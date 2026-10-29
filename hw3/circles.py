"""Подсчёт кружков на изображениях преобразованием Хафа.

Для каждого изображения подобраны свои параметры `cv2.HoughCircles`. Там, где
обычный `HOUGH_GRADIENT` даёт много ложных срабатываний на фоне (листва, блики,
повторяющиеся отверстия), используется более строгий `HOUGH_GRADIENT_ALT`.
Он требует почти идеальной окружности и поэтому чище отделяет настоящие торцы
и отверстия. Для снимка, где важно поймать больше мелких кружков, оставлен
обычный `HOUGH_GRADIENT`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Параметры подобраны вручную под каждый снимок.
# method: "gradient" — HOUGH_GRADIENT, "alt" — HOUGH_GRADIENT_ALT.
CONFIGS = {
    "logs-1.jpg": dict(method="alt", param2=0.60, min_dist=45, min_radius=15, max_radius=75),
    "logs-2.jpg": dict(method="alt", param2=0.60, min_dist=45, min_radius=15, max_radius=75),
    "logs-3.jpg": dict(method="alt", param2=0.65, min_dist=14, min_radius=6, max_radius=18),
    "logs-4.jpg": dict(method="alt", param2=0.55, min_dist=22, min_radius=12, max_radius=45),
    "pipes-4.jpg": dict(method="alt", param2=0.55, min_dist=20, min_radius=10, max_radius=26),
    "pipes-5.jpg": dict(method="gradient", param2=40, min_dist=45, min_radius=18, max_radius=60),
    "pipes-6.jpg": dict(method="alt", param2=0.60, min_dist=110, min_radius=50, max_radius=110),
}

DEFAULT = dict(method="gradient", param2=38, min_dist=30, min_radius=8, max_radius=60)

GRADIENT_PARAM1 = 120
GRADIENT_DP = 1.2
ALT_PARAM1 = 300
ALT_DP = 1.5
BLUR_KSIZE = 5


def detect_circles(gray: np.ndarray, config: dict) -> list[tuple[int, int, int]]:
    """Найти окружности на полутоновом изображении с заданными параметрами."""
    if config["method"] == "alt":
        blurred = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
        method, param1, dp = cv2.HOUGH_GRADIENT_ALT, ALT_PARAM1, ALT_DP
    else:
        blurred = cv2.medianBlur(gray, BLUR_KSIZE)
        method, param1, dp = cv2.HOUGH_GRADIENT, GRADIENT_PARAM1, GRADIENT_DP

    found = cv2.HoughCircles(
        blurred,
        method,
        dp=dp,
        minDist=config["min_dist"],
        param1=param1,
        param2=config["param2"],
        minRadius=config["min_radius"],
        maxRadius=config["max_radius"],
    )
    if found is None:
        return []
    return [(int(x), int(y), int(radius))
            for x, y, radius in np.round(found[0]).astype(int)]


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
        print(f"{path.name}: {len(circles)} кружков ({config['method']})")

    with (args.out / "counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "circles", "method", "param2", "min_dist",
                         "min_radius", "max_radius"])
        for name, count, config in rows:
            writer.writerow([name, count, config["method"], config["param2"],
                             config["min_dist"], config["min_radius"],
                             config["max_radius"]])


if __name__ == "__main__":
    main()
