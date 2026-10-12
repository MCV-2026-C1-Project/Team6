from pathlib import Path

import cv2
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Categorical

# =======================================================
# BEST COMBINATION FOUND
# =======================================================
# Color space: HSV
# Median blur fraction: 0.0000
# CLOSE fraction: 0.1032
# OPEN fraction: 0.4921
# Metric: error_rate
# Aggregation: mean
# Objective score: 0.089977
# Mean precision: 0.8416
# Mean recall: 0.8846
# Mean F1 score: 0.8611
# Mean error rate: 0.0900
# Worst error rate: 0.5844

# =======================================================
# IMAGE WITH THE HIGHEST ERROR
# =======================================================
# Image: 00020.jpg
# Error rate: 0.584411
# Error rate (%): 58.44%
# Error pixels: 403440
# Precision: 0.0000
# Recall: 0.0000
# F1 score: 0.0000
# =======================================================


# -------------------- CONFIGURATION --------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_DIR = DATA_DIR / "qsd2_w2"

BORDER_SIZE = 15

N_CALLS = 60
N_INITIAL_POINTS = 15
RANDOM_STATE = 42

# Fractions of the image's smaller dimension
OPEN_KERNEL_RANGE = (0.0, 1.0)
CLOSE_KERNEL_RANGE = (0.0, 1.0)
MEDIAN_BLUR_RANGE = (0.0, 0.1)

# Covariance regularization for Mahalanobis distance
COVARIANCE_REGULARIZATION = 1e-3

# "precision", "recall", "f1", "error_sum", "error_rate"
OBJECTIVE_METRIC = "error_rate"

# "mean" or "worst"
AGGREGATION = "mean"


# ------------------------------------------------------


def odd_kernel_size(min_dim, fraction, allow_zero=False):
    """Convert a size fraction into a valid odd kernel size."""
    if fraction <= 0:
        return 0 if allow_zero else 1

    size = max(3, int(min_dim * fraction))

    if size % 2 == 0:
        size += 1

    return size


def segment(
    image_bgr,
    color_space,
    open_fract,
    close_fract,
    median_blur_fraction,
):
    """
    Segment the foreground using Mahalanobis distance
    from the background color distribution estimated at the border.
    """

    color_space = color_space.upper()

    # 1. Convert image to the selected color space
    if color_space == "LAB":
        image = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2LAB
        ).astype(np.float32)

    elif color_space == "HSV":
        image = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2HSV
        ).astype(np.float32)

    else:
        raise ValueError(
            "color_space must be either 'HSV' or 'LAB'."
        )

    height, width = image.shape[:2]
    min_dim = min(height, width)

    # 2. Create border mask
    border = np.zeros((height, width), dtype=bool)

    b = max(
        1,
        min(BORDER_SIZE, height // 2, width // 2),
    )

    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True

    # 3. Estimate background color distribution
    background_pixels = image[border].astype(np.float64)
    mean = background_pixels.mean(axis=0)

    covariance = np.cov(background_pixels, rowvar=False)
    covariance += (
        np.eye(3) * COVARIANCE_REGULARIZATION
    )

    inverse_covariance = np.linalg.pinv(covariance)

    # 4. Calculate Mahalanobis distance
    delta = image.astype(np.float64) - mean

    distance_squared = np.einsum(
        "...i,ij,...j->...",
        delta,
        inverse_covariance,
        delta,
    )

    distance = np.sqrt(
        np.maximum(distance_squared, 0)
    )

    # Normalize distance for thresholding
    distance = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # 5. Threshold
    _, mask = cv2.threshold(
        distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
    )

    # 6. Median blur: kernel depends on image size
    median_size = odd_kernel_size(
        min_dim,
        median_blur_fraction,
        allow_zero=True,
    )

    # A size of zero disables median blur
    if median_size > 0:
        mask = cv2.medianBlur(mask, median_size)

    # 7. Calculate morphology kernel sizes
    close_size = odd_kernel_size(
        min_dim, close_fract
    )
    open_size = odd_kernel_size(
        min_dim, open_fract
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (close_size, close_size),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (open_size, open_size),
    )

    # 8. Apply morphology: CLOSE then OPEN
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    return mask


def calculate_metrics(mask, ground_truth):
    """Calculate binary segmentation metrics."""

    pred = mask > 0
    gt = ground_truth > 0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )
    recall = (
        tp / (tp + fn) if tp + fn else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    error_sum = fp + fn
    error_rate = error_sum / ground_truth.size

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "error_sum": int(error_sum),
        "error_rate": float(error_rate),
    }


def load_dataset():
    """Load images and their corresponding ground-truth masks."""

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    dataset = []
    missing_gt = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        gt_path = image_path.with_suffix(".png")

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"Warning: unable to read {image_path}")
            continue

        if gt is None:
            missing_gt.append(image_path.name)
            continue

        if gt.shape != image.shape[:2]:
            gt = cv2.resize(
                gt,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        dataset.append((image_path.name, image, gt))

    if missing_gt:
        print(
            f"Warning: {len(missing_gt)} images have no "
            f"ground truth. Skipping them: {missing_gt[:5]}"
        )

    if not dataset:
        raise RuntimeError(
            "No valid image and mask pairs were found."
        )

    print(f"Loaded {len(dataset)} images.")
    return dataset


def main():
    valid_metrics = {
        "precision",
        "recall",
        "f1",
        "error_sum",
        "error_rate",
    }

    if OBJECTIVE_METRIC not in valid_metrics:
        raise ValueError(
            f"OBJECTIVE_METRIC must be one of {valid_metrics}"
        )

    if AGGREGATION not in {"mean", "worst"}:
        raise ValueError(
            "AGGREGATION must be either 'mean' or 'worst'."
        )

    dataset = load_dataset()

    history = []
    evaluation = 0

    # Bayesian optimization search space
    space = [
        Categorical(["HSV", "LAB"], name="color_space"),
        Real(*OPEN_KERNEL_RANGE, name="open_fraction"),
        Real(*CLOSE_KERNEL_RANGE, name="close_fraction"),
        Real(
            *MEDIAN_BLUR_RANGE,
            name="median_blur_fraction",
        ),
    ]

    minimize_metric = OBJECTIVE_METRIC in {
        "error_sum",
        "error_rate",
    }

    def objective(params):
        nonlocal evaluation

        (
            color_space,
            open_fract,
            close_fract,
            median_blur_fraction,
        ) = params

        image_metrics = []
        per_image_results = []

        for image_name, image, gt in dataset:
            mask = segment(
                image,
                color_space,
                open_fract,
                close_fract,
                median_blur_fraction,
            )

            metrics = calculate_metrics(mask, gt)
            image_metrics.append(metrics)

            per_image_results.append({
                "image_name": image_name,
                **metrics,
            })

        metric_values = {
            key: np.array(
                [m[key] for m in image_metrics],
                dtype=float,
            )
            for key in valid_metrics
        }

        values = metric_values[OBJECTIVE_METRIC]

        if AGGREGATION == "mean":
            score = float(values.mean())
        elif minimize_metric:
            score = float(values.max())
        else:
            score = float(values.min())

        summary = {
            f"mean_{key}": float(vals.mean())
            for key, vals in metric_values.items()
        }

        summary.update({
            f"worst_{key}": float(
                vals.max()
                if key in {"error_sum", "error_rate"}
                else vals.min()
            )
            for key, vals in metric_values.items()
        })

        evaluation += 1

        history.append({
            "evaluation": evaluation,
            "color_space": color_space,
            "median_blur_fraction": median_blur_fraction,
            "close_fraction": close_fract,
            "open_fraction": open_fract,
            "objective_metric": OBJECTIVE_METRIC,
            "aggregation": AGGREGATION,
            "objective_score": score,
            "per_image_results": per_image_results,
            **summary,
        })

        print(
            f"{evaluation:03d}/{N_CALLS} | "
            f"color_space={color_space} | "
            f"median={median_blur_fraction:.4f} | "
            f"close={close_fract:.4f} | "
            f"open={open_fract:.4f} | "
            f"{OBJECTIVE_METRIC}={score:.6f} | "
            f"P={summary['mean_precision']:.4f}, "
            f"R={summary['mean_recall']:.4f}, "
            f"F1={summary['mean_f1']:.4f}, "
            f"Error={summary['mean_error_rate']:.4f}"
        )

        # gp_minimize always minimizes its objective
        return score if minimize_metric else -score

    # Run Bayesian optimization
    gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=N_CALLS,
        n_initial_points=min(N_INITIAL_POINTS, N_CALLS),
        random_state=RANDOM_STATE,
        acq_func="EI",
    )

    # Retrieve best evaluated combination
    if minimize_metric:
        best = min(
            history,
            key=lambda result: result["objective_score"],
        )
    else:
        best = max(
            history,
            key=lambda result: result["objective_score"],
        )

    print("\n" + "=" * 55)
    print("BEST COMBINATION FOUND")
    print("=" * 55)

    print(f"Color space: {best['color_space']}")
    print(
        "Median blur fraction: "
        f"{best['median_blur_fraction']:.4f}"
    )
    print(f"CLOSE fraction: {best['close_fraction']:.4f}")
    print(f"OPEN fraction: {best['open_fraction']:.4f}")
    print(f"Metric: {best['objective_metric']}")
    print(f"Aggregation: {best['aggregation']}")
    print(f"Objective score: {best['objective_score']:.6f}")
    print(f"Mean precision: {best['mean_precision']:.4f}")
    print(f"Mean recall: {best['mean_recall']:.4f}")
    print(f"Mean F1 score: {best['mean_f1']:.4f}")
    print(f"Mean error rate: {best['mean_error_rate']:.4f}")
    print(f"Worst error rate: {best['worst_error_rate']:.4f}")

    # Image with the highest error for the best parameters
    worst_image = max(
        best["per_image_results"],
        key=lambda result: result["error_rate"],
    )

    print("\n" + "=" * 55)
    print("IMAGE WITH THE HIGHEST ERROR")
    print("=" * 55)

    print(f"Image: {worst_image['image_name']}")
    print(f"Error rate: {worst_image['error_rate']:.6f}")
    print(
        f"Error rate (%): "
        f"{worst_image['error_rate'] * 100:.2f}%"
    )
    print(f"Error pixels: {worst_image['error_sum']}")
    print(f"Precision: {worst_image['precision']:.4f}")
    print(f"Recall: {worst_image['recall']:.4f}")
    print(f"F1 score: {worst_image['f1']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()