"""Сшивка кадров в панораму.

Для каждой группы снимков (`building1`, `building2`, `building3`) кадры
последовательно сопоставляются друг с другом:

1. на каждом кадре ищутся ключевые точки SIFT;
2. дескрипторы соседних кадров сопоставляются BFMatcher с проверкой отношения
   расстояний (Lowe ratio test);
3. по найденным соответствиям оценивается преобразование подобия
   (`estimateAffinePartial2D`: поворот, масштаб, сдвиг) с RANSAC;
4. преобразования пересчитываются в систему координат первого кадра, все кадры
   переводятся на общее полотно;
5. кадры смешиваются с весами по расстоянию до края (feathering), чтобы швы
   были незаметны.

Почему подобие, а не полная гомография. Фасады зданий состоят из повторяющихся
окон, и `findHomography` на таких соответствиях легко «уводит» перспективные
члены: углы кадра уезжают на десятки тысяч пикселей, а полотно разрастается до
неприемлемых размеров. Камера при съёмке движется примерно параллельно фасаду,
поэтому поворота, масштаба и сдвига достаточно; такая модель устойчива и не
даёт ложной перспективы.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Отношение расстояний для Lowe ratio test.
RATIO = 0.75
# Порог RANSAC при оценке преобразования, пиксели.
RANSAC_THRESH = 3.0
# Минимум соответствий, чтобы доверять преобразованию.
MIN_MATCHES = 10
# Предохранитель от абсурдно большого полотна.
MAX_PIXELS = 80_000_000


def _features(sift, image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def _transform(feat_a, feat_b):
    """Преобразование подобия, переводящее кадр b в систему координат кадра a."""
    kp_a, desc_a = feat_a
    kp_b, desc_b = feat_b
    if desc_a is None or desc_b is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_b, desc_a, k=2)
    good = [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]
    if len(good) < MIN_MATCHES:
        return None

    src = np.float32([kp_b[m.queryIdx].pt for m in good])
    dst = np.float32([kp_a[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
        confidence=0.999, maxIters=5000,
    )
    if matrix is None:
        return None
    return np.vstack([matrix, [0, 0, 1]]), len(good), int(inliers.sum())


def _corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def stitch(paths: list[Path]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Сшить кадры в панораму. Вернуть панораму и статистику по парам."""
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"не удалось прочитать файл: {path}")
        images.append(image)

    sift = cv2.SIFT_create()
    features = [_features(sift, image) for image in images]

    # Преобразования каждого кадра в систему координат первого.
    transforms = [np.eye(3, dtype=np.float64)]
    stats = []
    for i in range(1, len(images)):
        result = _transform(features[i - 1], features[i])
        if result is None:
            raise RuntimeError(
                f"не удалось сопоставить {paths[i - 1].name} и {paths[i].name}"
            )
        matrix, matches, inliers = result
        transforms.append(transforms[-1] @ matrix)
        stats.append((matches, inliers))

    # Границы общего полотна.
    projected = [cv2.perspectiveTransform(_corners(image.shape), matrix)
                 for image, matrix in zip(images, transforms)]
    points = np.concatenate(projected)
    xmin, ymin = points.min(axis=0).ravel()
    xmax, ymax = points.max(axis=0).ravel()
    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))
    if width * height > MAX_PIXELS:
        raise RuntimeError(f"полотно слишком большое: {width}x{height}")

    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)

    # Взвешенное смешивание по расстоянию до края кадра.
    accumulator = np.zeros((height, width, 3), np.float32)
    weights = np.zeros((height, width), np.float32)
    for image, matrix in zip(images, transforms):
        full = shift @ matrix
        warped = cv2.warpPerspective(image, full, (width, height))
        mask = cv2.warpPerspective(
            np.full(image.shape[:2], 255, np.uint8), full, (width, height),
            flags=cv2.INTER_NEAREST,
        )
        weight = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        weight += 1e-3
        accumulator += warped.astype(np.float32) * weight[..., None]
        weights += weight

    panorama = accumulator / np.maximum(weights, 1e-6)[..., None]
    panorama = panorama.clip(0, 255).astype(np.uint8)

    # Обрезать пустые поля.
    ys, xs = np.where(weights > 1e-3)
    if len(ys) and len(xs):
        panorama = panorama[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    return panorama, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Сшивка кадров в панораму.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data",
                        help="каталог с подкаталогами групп кадров")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output",
                        help="каталог для панорам")
    args = parser.parse_args()

    groups = sorted(p for p in args.data.iterdir() if p.is_dir())
    if not groups:
        raise SystemExit(f"в {args.data} нет каталогов с кадрами")

    args.out.mkdir(parents=True, exist_ok=True)
    for group in groups:
        paths = sorted(p for p in group.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if len(paths) < 2:
            continue

        panorama, stats = stitch(paths)
        out_path = args.out / f"{group.name}_panorama.jpg"
        cv2.imwrite(str(out_path), panorama, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"{group.name}: {len(paths)} кадров -> {panorama.shape[1]}x{panorama.shape[0]}")
        for i, (matches, inliers) in enumerate(stats, start=1):
            print(f"  пара {i - 1}-{i}: {matches} соответствий, {inliers} инлайеров")


if __name__ == "__main__":
    main()
