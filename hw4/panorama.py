"""Сшивка кадров в панораму.

Метод с практического занятия (lab04):

1. ORB (`nfeatures=2000`) находит ключевые точки и дескрипторы.
2. `BFMatcher` с `NORM_HAMMING` сопоставляет дескрипторы соседних кадров
   (`knnMatch`, k=2).
3. Пары фильтруются проверкой отношения расстояний (Lowe ratio test, 0.6).
4. По хорошим соответствиям оценивается гомография
   `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
5. Кадры переносятся на общее полотно через `cv2.warpPerspective` и
   смешиваются.

Для группы из нескольких кадров гомографии соседних пар перемножаются в
систему координат первого кадра.

Отдельно про устойчивость. Фасады зданий состоят из повторяющихся окон, и
полная гомография на таких соответствиях часто получает лишние перспективные
члены: у матрицы появляется полюс рядом с кадром, углы уезжают на десятки
тысяч пикселей, а полотно раздувается до гигабайтов. Поэтому после оценки
всей цепочки гомографий выполняется проверка её корректности. Если цепочка
неустойчива, группа сшивается преобразованием подобия (поворот, равномерный
масштаб, сдвиг) — это частный случай гомографии, и он оценивается по тем же
самым соответствиям через `cv2.estimateAffinePartial2D`. Такой вариант
устойчив и не даёт ложной перспективы.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Параметры из практического занятия.
ORB_FEATURES = 2000
RATIO = 0.6
RANSAC_THRESH = 5.0

# Минимум соответствий, чтобы доверять преобразованию.
MIN_MATCHES = 10
# Предохранитель от абсурдно большого полотна.
MAX_PIXELS = 80_000_000
# Допустимый разброс размера кадра после преобразования.
SPAN_LOW, SPAN_HIGH = 0.25, 5.0


def _features(orb, image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(gray, None)


def _good_matches(feat_a, feat_b):
    """Хорошие соответствия: для точек кадра b находятся точки кадра a."""
    _, desc_a = feat_a
    _, desc_b = feat_b
    if desc_a is None or desc_b is None:
        return []

    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_b, desc_a, k=2)
    return [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]


def _points(feat_a, feat_b, matches):
    kp_a, _ = feat_a
    kp_b, _ = feat_b
    src = np.float32([kp_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst


def _homography(feat_a, feat_b, matches):
    """Гомография, переводящая кадр b в систему координат кадра a."""
    if len(matches) < MIN_MATCHES:
        return None, 0
    src, dst = _points(feat_a, feat_b, matches)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    return matrix, 0 if inliers is None else int(inliers.sum())


def _similarity(feat_a, feat_b, matches):
    """Преобразование подобия b -> a на тех же соответствиях."""
    if len(matches) < MIN_MATCHES:
        return None, 0
    src, dst = _points(feat_a, feat_b, matches)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
        confidence=0.999, maxIters=5000,
    )
    if matrix is None:
        return None, 0
    return np.vstack([matrix, [0, 0, 1]]), int(inliers.sum())


def _corners(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def _sane(matrix: np.ndarray, shape) -> bool:
    """Проверить, что преобразование не раздувает кадр до абсурда."""
    if matrix is None or matrix.shape != (3, 3):
        return False
    projected = cv2.perspectiveTransform(_corners(shape), matrix).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False
    h, w = shape[:2]
    span_w = projected[:, 0].max() - projected[:, 0].min()
    span_h = projected[:, 1].max() - projected[:, 1].min()
    return (SPAN_LOW * w < span_w < SPAN_HIGH * w
            and SPAN_LOW * h < span_h < SPAN_HIGH * h)


def _plan(images, features):
    """Выбрать модель преобразования и построить матрицы к первому кадру."""
    pairs = []
    for i in range(1, len(images)):
        matches = _good_matches(features[i - 1], features[i])
        H, h_inliers = _homography(features[i - 1], features[i], matches)
        S, s_inliers = _similarity(features[i - 1], features[i], matches)
        pairs.append((len(matches), h_inliers, s_inliers, H, S))

    for model, index in (("homography", 3), ("similarity", 4)):
        transforms = [np.eye(3, dtype=np.float64)]
        valid = True
        for i, pair in enumerate(pairs, start=1):
            matrix = pair[index]
            if matrix is None:
                valid = False
                break
            transforms.append(transforms[-1] @ matrix)
            if not _sane(transforms[-1], images[i].shape):
                valid = False
                break
        if valid and _canvas_size(images, transforms) is not None:
            return model, transforms, pairs

    raise RuntimeError("не удалось подобрать устойчивое преобразование")


def _canvas_size(images, transforms):
    projected = [cv2.perspectiveTransform(_corners(image.shape), matrix)
                 for image, matrix in zip(images, transforms)]
    points = np.concatenate(projected)
    xmin, ymin = points.min(axis=0).ravel()
    xmax, ymax = points.max(axis=0).ravel()
    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))
    if width * height > MAX_PIXELS:
        return None
    return width, height, xmin, ymin


def stitch(paths: list[Path]):
    """Сшить кадры в панораму. Вернуть панораму, модель и статистику."""
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"не удалось прочитать файл: {path}")
        images.append(image)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    features = [_features(orb, image) for image in images]
    model, transforms, pairs = _plan(images, features)

    width, height, xmin, ymin = _canvas_size(images, transforms)
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

    ys, xs = np.where(weights > 1e-3)
    if len(ys) and len(xs):
        panorama = panorama[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    return panorama, model, pairs


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

        panorama, model, pairs = stitch(paths)
        out_path = args.out / f"{group.name}_panorama.jpg"
        cv2.imwrite(str(out_path), panorama, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"{group.name}: {len(paths)} кадров -> {panorama.shape[1]}x{panorama.shape[0]}"
              f" ({model})")
        for i, (matches, h_inliers, s_inliers, _, _) in enumerate(pairs, start=1):
            print(f"  пара {i - 1}-{i}: {matches} соответствий, "
                  f"гомография {h_inliers}, подобие {s_inliers}")


if __name__ == "__main__":
    main()
