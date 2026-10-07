"""Выделение границ почки на снимках УЗИ и КТ.

Для каждого снимка выполняется предварительная фильтрация (медианная или
гауссова), после чего применяются четыре оператора выделения границ:
Собель, Щарр, Прюитт и Кэнни. Итого 2 фильтра x 4 оператора = 8 результатов
на один снимок.

Функция `process(image)` принимает изображение и возвращает словарь с этими
восемью результатами.

Дополнительно каждый результат сравнивается с эталонной маской почки: контур
маски считается правильным ответом, и по нему измеряются precision, recall и
F1. Это даёт объективную оценку того, насколько чётко оператор выделяет именно
почку, а не шум и посторонние структуры.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Предварительная фильтрация.
MEDIAN_KSIZE = 5
GAUSS_SIGMA = 1.5

# Допуск совпадения границы с контуром маски, пиксели.
MATCH_TOL = 4

# Ядро Прюитта (в OpenCV готового оператора Прюитта нет).
PREWITT_X = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]], dtype=np.float32)
PREWITT_Y = PREWITT_X.T

FILTERS = ("median", "gaussian")
OPERATORS = ("sobel", "scharr", "prewitt", "canny")
COMBINATIONS = tuple(f"{f}_{o}" for f in FILTERS for o in OPERATORS)


def _binarize(magnitude: np.ndarray) -> np.ndarray:
    """Нормировать модуль градиента и разбить порогом Оцу."""
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _median(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, MEDIAN_KSIZE)


def _gaussian(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), GAUSS_SIGMA)


def sobel(image: np.ndarray) -> np.ndarray:
    """Оператор Собеля: модуль градиента."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return _binarize(cv2.magnitude(gx, gy))


def scharr(image: np.ndarray) -> np.ndarray:
    """Оператор Щарра: модуль градиента."""
    gx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    return _binarize(cv2.magnitude(gx, gy))


def prewitt(image: np.ndarray) -> np.ndarray:
    """Оператор Прюитта: модуль градиента."""
    gx = cv2.filter2D(image, cv2.CV_32F, PREWITT_X)
    gy = cv2.filter2D(image, cv2.CV_32F, PREWITT_Y)
    return _binarize(cv2.magnitude(gx, gy))


def canny(image: np.ndarray) -> np.ndarray:
    """Детектор Кэнни с порогами от медианы яркости."""
    med = float(np.median(image))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    return cv2.Canny(image, lo, hi)


OPERATOR_FUNCS = {
    "sobel": sobel,
    "scharr": scharr,
    "prewitt": prewitt,
    "canny": canny,
}
FILTER_FUNCS = {
    "median": _median,
    "gaussian": _gaussian,
}


def process(image: np.ndarray) -> dict[str, np.ndarray]:
    """Принять снимок, вернуть 8 карт границ: <фильтр>_<оператор>."""
    result = {}
    for filter_name, filter_func in FILTER_FUNCS.items():
        filtered = filter_func(image)
        for operator_name, operator_func in OPERATOR_FUNCS.items():
            result[f"{filter_name}_{operator_name}"] = operator_func(filtered)
    return result


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def evaluate(pred: np.ndarray, mask: np.ndarray, tol: int = MATCH_TOL):
    """Сравнить карту границ с контуром маски почки.

    Точность считается только по области вокруг почки: посторонние границы
    за её пределами не должны влиять на оценку.
    """
    truth = _mask_boundary(mask) > 0
    found = pred > 0
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    near_truth = cv2.dilate(_mask_boundary(mask), disk) > 0
    near_found = cv2.dilate(pred, disk) > 0
    roi = cv2.dilate(mask, disk) > 0

    found_roi = found & roi
    precision = float((found_roi & near_truth).sum()) / max(int(found_roi.sum()), 1)
    recall = float((truth & near_found).sum()) / max(int(truth.sum()), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def montage(mask: np.ndarray, results: dict[str, np.ndarray],
            f1: dict[str, float], title: str, path: Path) -> None:
    """Сетка 2x4 со всеми восемью результатами и контуром маски."""
    fig, axes = plt.subplots(len(FILTERS), len(OPERATORS), figsize=(16, 8))
    for i, filter_name in enumerate(FILTERS):
        for j, operator_name in enumerate(OPERATORS):
            name = f"{filter_name}_{operator_name}"
            ax = axes[i, j]
            ax.imshow(results[name], cmap="gray")
            ax.contour(mask, levels=[127], colors="red", linewidths=0.6)
            ax.set_title(f"{filter_name} + {operator_name}   F1 = {f1[name]:.2f}",
                         fontsize=10)
            ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_chart(mean_f1: dict[str, float], path: Path) -> None:
    """Столбчатая диаграмма среднего F1 по комбинациям."""
    names = list(mean_f1)
    values = [mean_f1[n] for n in names]
    colors = ["#4c72b0" if n.startswith("median") else "#dd8452" for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("средний F1")
    ax.set_ylim(0, 1)
    ax.axhline(max(values), color="green", linestyle="--", linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("Схожесть границы с контуром почки (среднее по 4 снимкам)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Границы почки операторами Собеля, Щарра, Прюитта и Кэнни.")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data",
                        help="каталог со снимками img_*.png и масками mask_*.png")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output",
                        help="каталог для результатов")
    args = parser.parse_args()

    images = sorted(args.data.glob("img_*.png"))
    if not images:
        raise SystemExit(f"в {args.data} нет файлов img_*.png")

    args.out.mkdir(parents=True, exist_ok=True)
    per_image = {name: [] for name in COMBINATIONS}
    rows = []

    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"не удалось прочитать файл: {path}")
        mask_path = args.data / path.name.replace("img_", "mask_")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"не удалось прочитать маску: {mask_path}")

        results = process(image)
        f1 = {}
        for name, edge in results.items():
            cv2.imwrite(str(args.out / f"{path.stem}_{name}.png"), edge)
            precision, recall, score = evaluate(edge, mask)
            f1[name] = score
            per_image[name].append(score)
            rows.append((path.stem, name, precision, recall, score))

        montage(mask, results, f1, path.stem, args.out / f"compare_{path.stem}.png")
        best = max(f1, key=f1.get)
        print(f"{path.name}: лучшая комбинация — {best} (F1 = {f1[best]:.3f})")

    mean_f1 = {name: float(np.mean(per_image[name])) for name in COMBINATIONS}
    summary_chart(mean_f1, args.out / "f1_by_combination.png")

    print("\nСредний F1 по комбинациям (4 снимка):")
    for name in COMBINATIONS:
        print(f"  {name:>16}: {mean_f1[name]:.3f}")

    best = max(mean_f1, key=mean_f1.get)
    print(f"\nЛучшая комбинация: {best} (F1 = {mean_f1[best]:.3f})")

    with (args.out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "combination", "precision", "recall", "f1"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])


if __name__ == "__main__":
    main()
