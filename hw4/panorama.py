"""Сшивка кадров в панораму — два метода.

1. ``lab04`` — метод с практического занятия:
   ORB -> BFMatcher с ``NORM_HAMMING`` -> Lowe ratio 0.6 ->
   ``findHomography`` (RANSAC 5.0) -> ``warpPerspective``.
   Если цепочка гомографий неустойчива, группа сшивается преобразованием
   подобия по тем же соответствиям.

2. ``sift`` — второй вариант:
   SIFT -> BFMatcher с ``NORM_L2`` -> Lowe ratio 0.75 ->
   ``estimateAffinePartial2D`` (подобие, RANSAC) -> ``warpPerspective``.

Оба метода смешивают кадры с весами по расстоянию до края кадра (feathering).
Опорным кадром выбирается центральный: ошибка накапливается в обе стороны от
него, поэтому панорама меньше заваливается и наклон уменьшается.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Параметры методов.
ORB_FEATURES = 2000
ORB_RATIO = 0.6
SIFT_RATIO = 0.75

# Общие параметры оценки преобразования.
RANSAC_THRESH = 5.0
MIN_MATCHES = 10

# Предохранитель от абсурдно большого полотна.
MAX_PIXELS = 80_000_000
# Допустимый разброс размера кадра после преобразования.
SPAN_LOW, SPAN_HIGH = 0.25, 5.0

METHODS = {
    "lab04": dict(detector="orb", norm=cv2.NORM_HAMMING, ratio=ORB_RATIO,
                  model="homography"),
    "sift": dict(detector="sift", norm=cv2.NORM_L2, ratio=SIFT_RATIO,
                 model="similarity"),
}


def _detector(name: str):
    if name == "orb":
        return cv2.ORB_create(nfeatures=ORB_FEATURES)
    if name == "sift":
        return cv2.SIFT_create()
    raise ValueError(f"неизвестный детектор: {name}")


def _features(detector, image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)


def _good_matches(feat_a, feat_b, norm: int, ratio: float):
    """Хорошие соответствия: для точек кадра b находятся точки кадра a."""
    _, desc_a = feat_a
    _, desc_b = feat_b
    if desc_a is None or desc_b is None:
        return []

    matcher = cv2.BFMatcher_create(norm)
    knn = matcher.knnMatch(desc_b, desc_a, k=2)
    return [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


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

    # Полотно не должно быть заметно больше суммы кадров: иначе цепочка
    # преобразований «разъехалась» и панорама растянута.
    total_width = sum(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    if width > 1.1 * total_width or height > 2.5 * max_height:
        return None
    return width, height, xmin, ymin


def _plan(images, features, config):
    """Выбрать модель преобразования и построить матрицы к первому кадру."""
    pairs = []
    for i in range(1, len(images)):
        matches = _good_matches(features[i - 1], features[i],
                                config["norm"], config["ratio"])
        H, h_inliers = _homography(features[i - 1], features[i], matches)
        S, s_inliers = _similarity(features[i - 1], features[i], matches)
        pairs.append(dict(matches=len(matches), H=H, h_inliers=h_inliers,
                          S=S, s_inliers=s_inliers))

    def chain(key: str):
        """Матрицы к опорному кадру. Опорный — центральный, ошибка идёт в обе стороны."""
        reference = len(images) // 2
        transforms = [None] * len(images)
        transforms[reference] = np.eye(3, dtype=np.float64)

        for i in range(reference + 1, len(images)):
            matrix = pairs[i - 1][key]
            if matrix is None:
                return None
            transforms[i] = transforms[i - 1] @ matrix
            if not _sane(transforms[i], images[i].shape):
                return None

        for i in range(reference - 1, -1, -1):
            matrix = pairs[i][key]  # переводит кадр i+1 в i
            if matrix is None:
                return None
            transforms[i] = transforms[i + 1] @ np.linalg.inv(matrix)
            if not _sane(transforms[i], images[i].shape):
                return None

        if _canvas_size(images, transforms) is None:
            return None
        return transforms

    if config["model"] == "homography":
        transforms = chain("H")
        if transforms is not None:
            return "homography", transforms, pairs

    transforms = chain("S")
    if transforms is not None:
        return "similarity", transforms, pairs
    raise RuntimeError("не удалось подобрать устойчивое преобразование")


def stitch(paths: list[Path], method: str):
    """Сшить кадры в панораму. Вернуть панораму, модель и статистику."""
    config = METHODS[method]
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(f"не удалось прочитать файл: {path}")
        images.append(image)

    detector = _detector(config["detector"])
    features = [_features(detector, image) for image in images]
    model, transforms, pairs = _plan(images, features, config)

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
    parser.add_argument("--method", choices=("both", "lab04", "sift"),
                        default="both", help="какой метод запускать")
    args = parser.parse_args()

    groups = sorted(p for p in args.data.iterdir() if p.is_dir())
    if not groups:
        raise SystemExit(f"в {args.data} нет каталогов с кадрами")

    selected = ("lab04", "sift") if args.method == "both" else (args.method,)
    for method in selected:
        out_dir = args.out / method
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {method} ===")
        for group in groups:
            paths = sorted(p for p in group.iterdir()
                           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
            if len(paths) < 2:
                continue

            panorama, model, pairs = stitch(paths, method)
            out_path = out_dir / f"{group.name}_panorama.jpg"
            cv2.imwrite(str(out_path), panorama, [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"{group.name}: {len(paths)} кадров -> "
                  f"{panorama.shape[1]}x{panorama.shape[0]} ({model})")
            for i, pair in enumerate(pairs, start=1):
                print(f"  пара {i - 1}-{i}: {pair['matches']} соответствий, "
                      f"гомография {pair['h_inliers']}, подобие {pair['s_inliers']}")


if __name__ == "__main__":
    main()
