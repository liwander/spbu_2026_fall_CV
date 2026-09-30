"""Определение цвета сигнала светофора по фотографии.

Метод без нейросетей, только классическая обработка:

1. Кадр приводится к фиксированному масштабу (чтобы размер линзы был
   более-менее одинаковым).
2. Изображение переводится в HSV.
3. Положение горящей линзы ищется через white top-hat по каналу яркости:
   линза светится и даёт локальный всплеск яркости, а плоский (даже яркий
   жёлтый) корпус такого всплеска не даёт.
4. Оттенок берётся по ярким насыщенным пикселям найденной области, с
   учётом заворота красного через 0/180.
5. Дополнительные правила: компактная зелёная или яркая красная область
   (фактически горящая линза) переопределяют результат. Это спасает кадры,
   где жёлтый корпус ярче самой линзы.

Синий и белый из анализа исключены: небо и блики не являются сигналом.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Диапазоны оттенков в OpenCV HSV (H: 0..180).
# Зелёный расширен до 100: бирюзовые линзы светофора дают H около 86-98.
RED_HI = 8
RED_LO = 165
YELLOW_LO, YELLOW_HI = 9, 35
GREEN_LO, GREEN_HI = 36, 100

# Масштаб, к которому приводится наибольшая сторона кадра.
WORK_SIZE = 256

# Параметры детектора свечения.
GLOW_KERNEL = 31
GLOW_MIN = 60
GLOW_MIN_SAT = 120
GLOW_MIN_ASPECT = 0.55
GLOW_WINDOW = 1.5

# Порог насыщенности: отсекает белые блики и серое небо.
SAT_MIN = 90

COLORS = ("red", "yellow", "green")


def _resize(image: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _in_red(h: np.ndarray) -> np.ndarray:
    return (h <= RED_HI) | (h >= RED_LO)


def _in_yellow(h: np.ndarray) -> np.ndarray:
    return (h >= YELLOW_LO) & (h <= YELLOW_HI)


def _in_green(h: np.ndarray) -> np.ndarray:
    return (h >= GREEN_LO) & (h <= GREEN_HI)


def _traffic(h: np.ndarray) -> np.ndarray:
    """Красный, жёлтый и зелёный. Небо и белые блики сюда не попадают."""
    return _in_red(h) | _in_yellow(h) | _in_green(h)


def _band_of(hue: int) -> str:
    if hue <= RED_HI or hue >= RED_LO:
        return "red"
    if hue <= YELLOW_HI:
        return "yellow"
    return "green"


def _peak_hue(h: np.ndarray, w: np.ndarray) -> str:
    """Доминирующий оттенок по взвешенной гистограмме (красный заворачивается)."""
    hist = np.bincount(h, weights=w.astype(np.float64), minlength=180)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    smooth = np.convolve(np.r_[hist, hist[:2]], kernel, "same")[:180]
    smooth[0] += smooth[178] + smooth[179] + smooth[177]
    smooth[179] += smooth[0] + smooth[1] + smooth[2]
    return _band_of(int(np.argmax(smooth)))


def _glow_label(hsv: np.ndarray) -> str | None:
    """Цвет линзы, найденной по локальному всплеску яркости (top-hat)."""
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2]

    traffic = _traffic(h) & (s >= GLOW_MIN_SAT)
    if traffic.sum() < 4:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GLOW_KERNEL, GLOW_KERNEL))
    glow = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel).astype(np.int32)
    candidate = traffic & (glow >= GLOW_MIN)
    if candidate.sum() < 4:
        return None

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    weight = np.bincount(labels.ravel(), weights=glow.ravel(), minlength=count)
    weight[0] = 0
    order = np.argsort(weight)[::-1]

    pick = next(
        (i for i in order
         if i != 0
         and min(stats[i, 2], stats[i, 3]) / max(stats[i, 2], stats[i, 3], 1) >= GLOW_MIN_ASPECT
         and stats[i, 4] >= 8),
        int(order[0]),
    )
    if pick == 0:
        return None

    _, _, bw, bh, _ = stats[pick]
    cx, cy = centroids[pick]
    half = max(GLOW_WINDOW * max(bw, bh) / 2, 8)
    yy, xx = np.ogrid[:h.shape[0], :h.shape[1]]
    window = (xx - cx) ** 2 + (yy - cy) ** 2 <= half ** 2

    selected = traffic & window & (v >= 0.55 * v[traffic].max())
    if selected.sum() < 4:
        selected = traffic & window
    if selected.sum() < 3:
        return None
    return _peak_hue(h[selected], s[selected] ** 2)


def _best_compact_blob(hsv: np.ndarray, hue_mask: np.ndarray):
    """Самая «дисковидная» связная область заданного цвета: (площадь, заполнение, яркость)."""
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.float32)

    mask = (hue_mask(h) & (s >= SAT_MIN)).astype(np.uint8)
    if mask.sum() < 12:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        area = stats[i, 4]
        if area < 12:
            continue
        fill = area / float(stats[i, 2] * stats[i, 3])
        score = fill * np.sqrt(area)
        if best is None or score > best[0]:
            best = (score, int(area), float(fill), float(v[labels == i].mean()))
    return None if best is None else best[1:]


def classify(path: Path) -> str:
    """Вернуть 'red', 'yellow' или 'green' для одного кадра."""
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"не удалось прочитать файл: {path}")
    hsv = cv2.cvtColor(_resize(image), cv2.COLOR_BGR2HSV)

    # Сначала — прямая проверка на компактную горящую линзу.
    green = _best_compact_blob(hsv, _in_green)
    if green is not None:
        area, fill, vmean = green
        if area >= 300 and fill >= 0.6 and vmean >= 80:
            return "green"

    red = _best_compact_blob(hsv, _in_red)
    if red is not None:
        area, fill, vmean = red
        if area >= 300 and fill >= 0.5 and vmean >= 120:
            return "red"

    return _glow_label(hsv) or "yellow"


def main() -> None:
    parser = argparse.ArgumentParser(description="Цвет сигнала светофора на фотографии.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data",
                        help="каталог с изображениями")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent,
                        help="каталог для green.txt / yellow.txt / red.txt")
    args = parser.parse_args()

    files = sorted(p for p in args.data.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})

    result = {color: [] for color in COLORS}
    for path in files:
        color = classify(path)
        result[color].append(path.name)
        print(f"{path.name} -> {color}")

    args.out.mkdir(parents=True, exist_ok=True)
    for color, names in result.items():
        (args.out / f"{color}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        print(f"{color}.txt: {len(names)}")


if __name__ == "__main__":
    main()
