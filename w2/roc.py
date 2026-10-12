from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------- CONFIGURATION --------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_DIR = DATA_DIR / "qsd2_w2"

BORDER_SIZE = 15
COVARIANCE_REGULARIZATION = 1e-3

# Choose "HSV" or "LAB"
COLOR_SPACE = "LAB"

# Manual thresholds to evaluate
THRESHOLDS = np.arange(0, 256, 1)

# ------------------------------------------------------


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

        dataset.append((image_path.name, image, gt > 127))

    if missing_gt:
        print(
            f"Warning: {len(missing_gt)} images have no "
            f"ground truth. Skipping: {missing_gt[:5]}"
        )

    if not dataset:
        raise RuntimeError(
            "No valid image and ground-truth pairs found."
        )

    print(f"Loaded {len(dataset)} images.")
    return dataset


def calculate_mahalanobis_distance(image_bgr, color_space):
    """
    Calculate Mahalanobis distance from the background
    distribution estimated using the image border.

    Returns a distance map normalized to [0, 255].
    No thresholding or filtering is applied here.
    """

    color_space = color_space.upper()

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
            "COLOR_SPACE must be 'HSV' or 'LAB'."
        )

    height, width = image.shape[:2]

    # Create border mask
    border = np.zeros((height, width), dtype=bool)

    b = max(
        1,
        min(BORDER_SIZE, height // 2, width // 2),
    )

    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True

    # Estimate background distribution
    background_pixels = image[border].astype(np.float64)
    mean = background_pixels.mean(axis=0)

    covariance = np.cov(
        background_pixels,
        rowvar=False,
    )

    covariance += (
        np.eye(3) * COVARIANCE_REGULARIZATION
    )

    inverse_covariance = np.linalg.pinv(covariance)

    # Mahalanobis distance for each pixel
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

    # Normalize to [0, 255] so manual thresholds are comparable
    distance = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    return distance


def evaluate_threshold(dataset, threshold):
    """Evaluate one threshold over the entire dataset."""

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _, distance, gt in dataset:
        # Foreground = distance above the threshold
        pred = distance > threshold

        tp += np.logical_and(pred, gt).sum()
        fp += np.logical_and(pred, ~gt).sum()
        tn += np.logical_and(~pred, ~gt).sum()
        fn += np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    youden_j = recall - fpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": int(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "youden_j": float(youden_j),
        "f1": float(f1),
    }


def main():
    print(f"Color space: {COLOR_SPACE}")
    print("Calculating Mahalanobis distance maps...")

    raw_dataset = load_dataset()

    # Calculate distance maps once; reuse them for every threshold.
    distance_dataset = []

    for image_name, image, gt in raw_dataset:
        distance = calculate_mahalanobis_distance(
            image,
            COLOR_SPACE,
        )

        distance_dataset.append(
            (image_name, distance, gt)
        )

    print(f"Evaluating {len(THRESHOLDS)} thresholds...")

    results = [
        evaluate_threshold(distance_dataset, threshold)
        for threshold in THRESHOLDS
    ]

    # Best threshold = maximum Youden's J
    best = max(
        results,
        key=lambda result: result["youden_j"],
    )

    print("\n" + "=" * 55)
    print("BEST MANUAL THRESHOLD")
    print("=" * 55)
    print(f"Color space: {COLOR_SPACE}")
    print(f"Threshold: {best['threshold']}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall / TPR: {best['recall']:.4f}")
    print(f"FPR: {best['fpr']:.4f}")
    print(f"Specificity: {best['specificity']:.4f}")
    print(f"F1 score: {best['f1']:.4f}")
    print(f"Youden's J: {best['youden_j']:.4f}")
    print(f"TP: {best['tp']}")
    print(f"FP: {best['fp']}")
    print(f"TN: {best['tn']}")
    print(f"FN: {best['fn']}")
    print("=" * 55)

    # Print nearby thresholds for comparison
    print("\nThreshold comparison around the best value:")
    print(
        f"{'Thr':>5} {'Precision':>11} {'Recall':>10} "
        f"{'FPR':>10} {'F1':>10} {'Youden J':>10}"
    )

    nearby = [
        result for result in results
        if abs(result["threshold"] - best["threshold"]) <= 5
    ]

    for result in nearby:
        marker = "*" if result["threshold"] == best["threshold"] else " "

        print(
            f"{marker}{result['threshold']:4d} "
            f"{result['precision']:11.4f} "
            f"{result['recall']:10.4f} "
            f"{result['fpr']:10.4f} "
            f"{result['f1']:10.4f} "
            f"{result['youden_j']:10.4f}"
        )

    # -------------------- ROC CURVE --------------------

    fpr_values = np.array(
        [result["fpr"] for result in results]
    )
    tpr_values = np.array(
        [result["recall"] for result in results]
    )

    plt.figure(figsize=(9, 7))

    plt.plot(
        fpr_values,
        tpr_values,
        label=f"ROC ({COLOR_SPACE})",
        linewidth=2,
    )

    # Random-classifier reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    # Mark the best threshold
    plt.scatter(
        [best["fpr"]],
        [best["recall"]],
        s=100,
        marker="o",
        label=(
            f"Best threshold = {best['threshold']}\n"
            f"P={best['precision']:.3f}, "
            f"R={best['recall']:.3f}"
        ),
        zorder=5,
    )

    plt.annotate(
        f"thr={best['threshold']}\n"
        f"P={best['precision']:.3f}\n"
        f"R={best['recall']:.3f}",
        (best["fpr"], best["recall"]),
        xytext=(12, -35),
        textcoords="offset points",
    )

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title(
        f"ROC curve before median blur and morphology ({COLOR_SPACE})"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path("roc_curve.png")
    plt.savefig(output_path, dpi=200)
    print(f"\nROC curve saved to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()