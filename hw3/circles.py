"""Подсчёт кружков на изображениях преобразованием Хафа.

Для каждого изображения подобраны свои параметры `cv2.HoughCircles`:
радиус искомых окружностей, минимальное расстояние между центрами, порог
аккумулятора и, где нужно, минимальная средняя яркость внутри кружка.
Яркостный фильтр отсекает ложные окружности на тёмной листве: торцы брёвен
светлые, а кроны деревьев — нет.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Параметры подобраны вручную под каждый снимок.
# min_brightness=None — фильтр по яркости не применяется.
CONFIGS = {
    "logs-1.jpg": dict(param2=38, min_dist=45, min_radius=15, max_radius=75, min_brightness=115),
    "logs-2.jpg": dict(param2=38, min_dist=45, min_radius=15, max_radius=75, min_brightness=115),
    "logs-3.jpg": dict(param2=28, min_dist=14, min_radius=6, max_radius=18, min_brightness=None),
    "logs-4.jpg": dict(param2=30, min_dist=22, min_radius=12, max_radius=45, min_brightness=110),
    "pipes-4.jpg": dict(param2=28, min_dist=20, min_radius=10, max_radius=26, min_brightness=None),
    "pipes-5.jpg": dict(param2=40, min_dist=45, min_radius=18, max_radius=60, min_brightness=None),
    "pipes-6.jpg": dict(param2=45, min_dist=110, min_radius=50, max_radius=110, min_brightness=None),
}

DEFAULT = dict(param2=38, min_dist=30, min_radius=8, max_radius=60, min_brightness=None)
PARAM1 = 120
DP = 1.2
BLUR_KSIZE = 5


def detect_circles(gray: np.ndarray, config: dict) -> list[tuple[int, int, int]]:
    """Найти окружности на полутоновом изображении с заданными параметрами."""
    blurred = cv2.medianBlur(gray, BLUR_KSIZE)
    found = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=config["min_dist"],
        param1=PARAM1,
        param2=config["param2"],
        minRadius=config["min_radius"],
        maxRadius=config["max_radius"],
    )
    if found is None:
        return []

    circles = []
    for x, y, radius in np.round(found[0]).astype(int):
        threshold = config["min_brightness"]
        if threshold is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), max(2, int(radius * 0.7)), 255, -1)
            if cv2.mean(gray, mask=mask)[0] < threshold:
                continue
        circles.append((int(x), int(y), int(radius)))
    return circles


def draw_circles(image: np.ndarray, circles: list[tuple[int, int, int]]) -> np.ndarray:
    """Обвести найденные окружности и отметить центры."""
    out = image.copy()
    for x, y, radius in circles:
        cv2.circle(out, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Подсчёт кружков преобразованием Хафа.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data",
                        help="каталог с изображениями")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output",
                        help="каталог для размеченных изображений")
    args = parser.parse_args()

    files = sorted(p for p in args.data.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not files:
        raise SystemExit(f"в {args.data} нет изображений")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"не удалось прочитать файл: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        config = CONFIGS.get(path.name, DEFAULT)
        circles = detect_circles(gray, config)

        cv2.imwrite(str(args.out / f"{path.stem}_circles.jpg"), draw_circles(image, circles))
        rows.append((path.name, len(circles), config))
        print(f"{path.name}: {len(circles)} кружков")

    with (args.out / "counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "circles", "param2", "min_dist", "min_radius",
                         "max_radius", "min_brightness"])
        for name, count, config in rows:
            writer.writerow([name, count, config["param2"], config["min_dist"],
                             config["min_radius"], config["max_radius"],
                             config["min_brightness"]])


if __name__ == "__main__":
    main()
