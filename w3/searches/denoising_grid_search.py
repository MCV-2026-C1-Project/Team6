"""Compare linear and nonlinear denoising filters on QSD1-W3.

The script:
1. Matches each QSD1-W3 image with its clean reference by numeric image ID.
2. Flags noisy development images using impulse and paired-reference checks.
3. Applies each candidate filter only to flagged images.
4. Compares each result with the clean reference using PSNR and SSIM.
5. Saves CSV tables, slide notes, charts, and visual examples.

The paired clean references are used for development analysis and scoring.
The additive-noise estimate uses those references, so it is not a
clean-reference-free detector for future test images.
"""

from pathlib import Path
import csv
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# -----------------------------------------------------------------------------
# Paths and experiment settings
# -----------------------------------------------------------------------------

# This script is in w3/searches/, so parents[2] points to the repository root.
ROOT_DIR = Path(__file__).resolve().parents[2]

QSD1_DIR = ROOT_DIR / "data" / "qsd1_w3"
CLEAN_DIR = QSD1_DIR / "non_augmented"
RESULTS_DIR = ROOT_DIR / "w3" / "searches" / "results" / "denoising"

# Noise detection thresholds used for this development-set experiment.
IMPULSE_FRACTION_THRESHOLD = 0.01
IMPULSE_INTENSITY_DIFFERENCE = 30
ADDITIVE_NOISE_SIGMA_THRESHOLD = 8.0

# Candidate filters. Mean and Gaussian filters are linear; median filters
# are nonlinear. No filter is the baseline for comparison.
FILTERS = [
    ("no_filter", "No filter", "Baseline"),
    ("mean_3", "Mean 3x3", "Linear"),
    ("mean_5", "Mean 5x5", "Linear"),
    ("gaussian_3", "Gaussian 3x3", "Linear"),
    ("gaussian_5", "Gaussian 5x5", "Linear"),
    ("median_3", "Median 3x3", "Nonlinear"),
    ("median_5", "Median 5x5", "Nonlinear"),
]

# Consistent colors make the charts easier to read.
COLORS = {
    "Baseline": "#8A8F98",
    "Linear": "#3977B8",
    "Nonlinear": "#D9822B",
    "Median 3x3": "#168A68",
    "Flagged": "#D9534F",
    "Not flagged": "#BCC3CC",
}


# -----------------------------------------------------------------------------
# Loading and ID matching
# -----------------------------------------------------------------------------

def image_id_from_path(path):
    """Return the numeric image ID from a filename such as 00005.jpg."""
    return int(path.stem.split("_")[-1])


def load_images_by_id(folder):
    """Load JPG images into a dictionary keyed by their numeric IDs.

    Args:
        folder: Dataset folder containing JPG files.

    Returns:
        Dictionary mapping image ID to an image loaded in OpenCV BGR format.

    Raises:
        FileNotFoundError: If the folder or JPG images are missing.
        ValueError: If an image cannot be read or an ID appears twice.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths = sorted(folder.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in: {folder}")

    images = {}
    for path in image_paths:
        image_id = image_id_from_path(path)
        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Could not read image: {path}")
        if image_id in images:
            raise ValueError(f"Duplicate image ID {image_id} in {folder}")

        images[image_id] = image

    return images


def load_matched_pairs():
    """Load QSD1-W3 images and their clean references, matched by image ID.

    Returns:
        A sorted list of dictionaries containing `image_id`, `noisy`, and
        `clean` image arrays.
    """
    noisy_images = load_images_by_id(QSD1_DIR)
    clean_images = load_images_by_id(CLEAN_DIR)

    matched_ids = sorted(set(noisy_images) & set(clean_images))
    if not matched_ids:
        raise ValueError(
            f"No matching image IDs between {QSD1_DIR} and {CLEAN_DIR}"
        )

    missing_clean = sorted(set(noisy_images) - set(clean_images))
    if missing_clean:
        raise ValueError(
            "Some QSD1-W3 images have no matching clean reference: "
            f"{missing_clean}"
        )

    pairs = []
    for image_id in matched_ids:
        noisy = noisy_images[image_id]
        clean = clean_images[image_id]

        if noisy.shape != clean.shape:
            raise ValueError(
                f"Image {image_id} has mismatched shapes: "
                f"QSD1={noisy.shape}, clean={clean.shape}"
            )

        pairs.append({
            "image_id": image_id,
            "noisy": noisy,
            "clean": clean,
        })

    return pairs


# -----------------------------------------------------------------------------
# Noise detection
# -----------------------------------------------------------------------------

def impulse_noise_fraction(image_bgr):
    """Estimate the fraction of isolated pixels that look impulse-like.

    A grayscale pixel counts as isolated if it differs by more than the
    configured intensity threshold from each of its eight surrounding pixels.
    The result is a fraction from 0 to 1.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    height, width = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    isolated = np.ones((height, width), dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue

            neighbor = padded[
                1 + dy:1 + dy + height,
                1 + dx:1 + dx + width,
            ]
            isolated &= (
                np.abs(gray - neighbor) > IMPULSE_INTENSITY_DIFFERENCE
            )

    return float(isolated.mean())


def estimate_residual_sigma(noisy_bgr, clean_bgr):
    """Estimate noise strength from a matched noisy/clean image pair.

    The median absolute deviation (MAD) is robust to large outlier pixels.
    Dividing by 0.6745 converts MAD to an approximate standard deviation for
    Gaussian-like residual noise.

    This estimate requires a clean reference, so it is for development-set
    analysis only.
    """
    noisy_gray = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    residual = noisy_gray - clean_gray
    residual_center = np.median(residual)
    mad = np.median(np.abs(residual - residual_center))

    return float(mad / 0.6745)


def detect_noisy_pairs(pairs):
    """Measure noise indicators and flag pairs that pass either threshold.

    An image is flagged when its impulse-like pixel fraction exceeds 1% or
    its paired-reference residual sigma exceeds the configured threshold.
    """
    detections = []

    for pair in pairs:
        impulse_fraction = impulse_noise_fraction(pair["noisy"])
        residual_sigma = estimate_residual_sigma(pair["noisy"], pair["clean"])

        impulse_flag = impulse_fraction > IMPULSE_FRACTION_THRESHOLD
        additive_flag = residual_sigma > ADDITIVE_NOISE_SIGMA_THRESHOLD

        detections.append({
            "image_id": pair["image_id"],
            "impulse_fraction": impulse_fraction,
            "residual_sigma": residual_sigma,
            "impulse_flag": impulse_flag,
            "additive_flag": additive_flag,
            "noisy_flag": impulse_flag or additive_flag,
        })

    return detections


# -----------------------------------------------------------------------------
# Filters and image-quality metrics
# -----------------------------------------------------------------------------

def apply_filter(image_bgr, filter_key):
    """Apply one candidate filter and return a new BGR image."""
    if filter_key == "no_filter":
        return image_bgr.copy()
    if filter_key == "mean_3":
        return cv2.blur(image_bgr, (3, 3))
    if filter_key == "mean_5":
        return cv2.blur(image_bgr, (5, 5))
    if filter_key == "gaussian_3":
        return cv2.GaussianBlur(image_bgr, (3, 3), 0)
    if filter_key == "gaussian_5":
        return cv2.GaussianBlur(image_bgr, (5, 5), 0)
    if filter_key == "median_3":
        return cv2.medianBlur(image_bgr, 3)
    if filter_key == "median_5":
        return cv2.medianBlur(image_bgr, 5)

    raise ValueError(f"Unknown filter: {filter_key}")


def calculate_psnr_and_ssim(predicted_bgr, clean_bgr):
    """Calculate PSNR in dB and color SSIM against the clean reference."""
    psnr = peak_signal_noise_ratio(
        clean_bgr,
        predicted_bgr,
        data_range=255,
    )

    # SSIM requires an odd window size no larger than the image dimensions.
    smallest_dimension = min(clean_bgr.shape[:2])
    win_size = min(7, smallest_dimension)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(
            "Images must be at least 3 pixels wide and tall for SSIM."
        )

    ssim = structural_similarity(
        clean_bgr,
        predicted_bgr,
        data_range=255,
        channel_axis=2,
        win_size=win_size,
    )

    return float(psnr), float(ssim)


def evaluate_filters(pairs, detections):
    """Run each filter on flagged images and measure quality per image.

    Unflagged images are excluded from the filter comparison. This keeps
    unchanged clean-looking images from dominating the average scores.
    """
    detection_by_id = {
        row["image_id"]: row
        for row in detections
    }
    flagged_pairs = [
        pair for pair in pairs
        if detection_by_id[pair["image_id"]]["noisy_flag"]
    ]

    if not flagged_pairs:
        raise ValueError("The noise detectors did not flag any images.")

    per_image_rows = []

    for pair in flagged_pairs:
        image_id = pair["image_id"]
        clean = pair["clean"]

        for filter_key, label, filter_type in FILTERS:
            if filter_key == "no_filter":
                result = pair["noisy"]
            else:
                result = apply_filter(pair["noisy"], filter_key)

            psnr, ssim = calculate_psnr_and_ssim(result, clean)
            per_image_rows.append({
                "image_id": image_id,
                "filter_key": filter_key,
                "method": label,
                "filter_type": filter_type,
                "psnr_db": psnr,
                "ssim": ssim,
            })

    return flagged_pairs, per_image_rows


def summarize_metrics(per_image_rows):
    """Summarize mean, standard deviation, and wins for each filter."""
    baseline_rows = {
        row["image_id"]: row
        for row in per_image_rows
        if row["filter_key"] == "no_filter"
    }

    summary = []
    for filter_key, label, filter_type in FILTERS:
        rows = [
            row for row in per_image_rows
            if row["filter_key"] == filter_key
        ]

        psnr_values = np.array([row["psnr_db"] for row in rows], dtype=float)
        ssim_values = np.array([row["ssim"] for row in rows], dtype=float)

        psnr_wins = 0
        ssim_wins = 0
        for row in rows:
            baseline = baseline_rows[row["image_id"]]
            psnr_wins += row["psnr_db"] > baseline["psnr_db"]
            ssim_wins += row["ssim"] > baseline["ssim"]

        baseline_psnr = np.mean([
            row["psnr_db"] for row in baseline_rows.values()
        ])
        baseline_ssim = np.mean([
            row["ssim"] for row in baseline_rows.values()
        ])

        summary.append({
            "filter_key": filter_key,
            "method": label,
            "filter_type": filter_type,
            "mean_psnr_db": float(np.mean(psnr_values)),
            "std_psnr_db": float(np.std(psnr_values)),
            "mean_ssim": float(np.mean(ssim_values)),
            "std_ssim": float(np.std(ssim_values)),
            "images_evaluated": len(rows),
            "psnr_wins_vs_no_filter": int(psnr_wins),
            "ssim_wins_vs_no_filter": int(ssim_wins),
            "delta_mean_psnr_db": float(np.mean(psnr_values) - baseline_psnr),
            "delta_mean_ssim": float(np.mean(ssim_values) - baseline_ssim),
        })

    return summary


# -----------------------------------------------------------------------------
# CSV and written report outputs
# -----------------------------------------------------------------------------

def write_csv(path, rows, fieldnames):
    """Write a list of dictionaries as a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_csv_outputs(detections, per_image_rows, summary):
    """Save noise-detection details and filter metrics as CSV files."""
    write_csv(
        RESULTS_DIR / "noise_detection.csv",
        detections,
        [
            "image_id",
            "impulse_fraction",
            "residual_sigma",
            "impulse_flag",
            "additive_flag",
            "noisy_flag",
        ],
    )

    write_csv(
        RESULTS_DIR / "filter_metrics_per_image.csv",
        per_image_rows,
        [
            "image_id",
            "filter_key",
            "method",
            "filter_type",
            "psnr_db",
            "ssim",
        ],
    )

    write_csv(
        RESULTS_DIR / "filter_metrics_summary.csv",
        summary,
        [
            "filter_key",
            "method",
            "filter_type",
            "mean_psnr_db",
            "std_psnr_db",
            "mean_ssim",
            "std_ssim",
            "images_evaluated",
            "psnr_wins_vs_no_filter",
            "ssim_wins_vs_no_filter",
            "delta_mean_psnr_db",
            "delta_mean_ssim",
        ],
    )


def save_slide_notes(detections, summary):
    """Write a short Markdown explanation with the experiment results."""
    flagged_ids = [
        row["image_id"] for row in detections if row["noisy_flag"]
    ]
    best_psnr = max(summary, key=lambda row: row["mean_psnr_db"])
    best_ssim = max(summary, key=lambda row: row["mean_ssim"])

    lines = [
        "# Denoising experiment summary",
        "",
        "## Experiment",
        "",
        f"- Matched QSD1-W3 images: **{len(detections)}**.",
        f"- Flagged noisy IDs: **{flagged_ids}** ({len(flagged_ids)} images).",
        "- Filters were evaluated only on flagged images.",
        "- Quality was measured against each image's matched clean reference.",
        "- Metrics: PSNR (higher is better) and SSIM (higher is better).",
        "",
        "## Best results",
        "",
        f"- Highest mean PSNR: **{best_psnr['method']}** "
        f"({best_psnr['mean_psnr_db']:.2f} dB).",
        f"- Highest mean SSIM: **{best_ssim['method']}** "
        f"({best_ssim['mean_ssim']:.3f}).",
        "",
        "## Comparison table",
        "",
        "| Type | Filter | Mean PSNR (dB) | Δ PSNR | Mean SSIM | Δ SSIM | PSNR wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        lines.append(
            f"| {row['filter_type']} | {row['method']} "
            f"| {row['mean_psnr_db']:.2f} "
            f"| {row['delta_mean_psnr_db']:+.2f} "
            f"| {row['mean_ssim']:.3f} "
            f"| {row['delta_mean_ssim']:+.3f} "
            f"| {row['psnr_wins_vs_no_filter']}/"
            f"{row['images_evaluated']} |"
        )

    lines.extend([
        "",
        "## Interpretation and limitation",
        "",
        "The paired clean references make PSNR and SSIM comparisons possible.",
        "The residual-sigma noise check also uses those references; it is a",
        "development-set analysis step, not a detector that can be applied",
        "to future images when their clean references are unavailable.",
        "",
    ])

    (RESULTS_DIR / "denoising_summary_for_slides.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def to_rgb(image_bgr):
    """Convert an OpenCV BGR image to RGB for Matplotlib display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_detection_chart(detections):
    """Plot both noise signals for all queries and mark flagged images."""
    ids = [row["image_id"] for row in detections]
    positions = np.arange(len(ids))
    colors = [
        COLORS["Flagged"] if row["noisy_flag"] else COLORS["Not flagged"]
        for row in detections
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(
        positions,
        [row["impulse_fraction"] * 100 for row in detections],
        color=colors,
    )
    axes[0].axhline(
        IMPULSE_FRACTION_THRESHOLD * 100,
        color="black",
        linestyle="--",
        label=f"Threshold: {IMPULSE_FRACTION_THRESHOLD * 100:.1f}%",
    )
    axes[0].set_ylabel("Impulse-like pixels (%)")
    axes[0].set_title("Noise detection signals across QSD1-W3")
    axes[0].legend()

    axes[1].bar(
        positions,
        [row["residual_sigma"] for row in detections],
        color=colors,
    )
    axes[1].axhline(
        ADDITIVE_NOISE_SIGMA_THRESHOLD,
        color="black",
        linestyle="--",
        label=f"Threshold: {ADDITIVE_NOISE_SIGMA_THRESHOLD:.1f}",
    )
    axes[1].set_ylabel("Paired residual sigma")
    axes[1].set_xlabel("QSD1-W3 image ID")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(ids, rotation=45)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_noise_detection.png", dpi=180)
    plt.close(fig)


def save_filter_scorecard(summary):
    """Compare average PSNR and SSIM, with standard-deviation error bars."""
    labels = [row["method"] for row in summary]
    positions = np.arange(len(summary))
    colors = [
        COLORS["Baseline"] if row["filter_type"] == "Baseline"
        else COLORS["Nonlinear"] if row["method"] == "Median 3x3"
        else COLORS[row["filter_type"]]
        for row in summary
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(
        positions,
        [row["mean_psnr_db"] for row in summary],
        yerr=[row["std_psnr_db"] for row in summary],
        capsize=4,
        color=colors,
    )
    axes[0].set_title("PSNR against clean references")
    axes[0].set_ylabel("Mean PSNR (dB)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        positions,
        [row["mean_ssim"] for row in summary],
        yerr=[row["std_ssim"] for row in summary],
        capsize=4,
        color=colors,
    )
    axes[1].set_title("SSIM against clean references")
    axes[1].set_ylabel("Mean SSIM")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Filter comparison on detected noisy images "
        "(bars show mean ± standard deviation)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_filter_scorecard.png", dpi=180)
    plt.close(fig)


def save_ssim_gain_heatmap(flagged_pairs, per_image_rows):
    """Show each filter's SSIM gain over no filter for every noisy image."""
    image_ids = [pair["image_id"] for pair in flagged_pairs]
    filter_labels = [label for _, label, _ in FILTERS]

    values = np.zeros((len(image_ids), len(FILTERS)), dtype=float)
    lookup = {
        (row["image_id"], row["filter_key"]): row["ssim"]
        for row in per_image_rows
    }

    for image_index, image_id in enumerate(image_ids):
        baseline = lookup[(image_id, "no_filter")]
        for filter_index, (filter_key, _, _) in enumerate(FILTERS):
            values[image_index, filter_index] = (
                lookup[(image_id, filter_key)] - baseline
            )

    limit = max(float(np.max(np.abs(values))), 0.001)

    fig, ax = plt.subplots(
        figsize=(12, max(4, 0.65 * len(image_ids) + 2))
    )
    image = ax.imshow(
        values,
        cmap="RdYlGn",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(filter_labels)))
    ax.set_xticklabels(filter_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(image_ids)))
    ax.set_yticklabels([f"ID {image_id}" for image_id in image_ids])
    ax.set_title("Per-image SSIM change compared with no filter")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("SSIM gain (green = improvement)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_per_image_ssim_heatmap.png", dpi=180)
    plt.close(fig)


def save_metric_tradeoff(summary):
    """Plot each filter in PSNR/SSIM space to show the two-metric trade-off."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for row in summary:
        color = (
            COLORS["Baseline"] if row["filter_type"] == "Baseline"
            else COLORS["Nonlinear"] if row["filter_type"] == "Nonlinear"
            else COLORS["Linear"]
        )
        size = 180 if row["method"] == "Median 3x3" else 90

        ax.scatter(
            row["mean_psnr_db"],
            row["mean_ssim"],
            color=color,
            s=size,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            row["method"],
            (row["mean_psnr_db"], row["mean_ssim"]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Filter quality map: PSNR and SSIM together")
    ax.set_xlabel("Mean PSNR (dB), higher is better")
    ax.set_ylabel("Mean SSIM, higher is better")
    ax.grid(alpha=0.25)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label="Baseline",
            markerfacecolor=COLORS["Baseline"], markeredgecolor="black",
            markersize=9,
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", label="Linear",
            markerfacecolor=COLORS["Linear"], markeredgecolor="black",
            markersize=9,
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", label="Nonlinear",
            markerfacecolor=COLORS["Nonlinear"], markeredgecolor="black",
            markersize=9,
        ),
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_psnr_ssim_quality_map.png", dpi=180)
    plt.close(fig)


def choose_example_ids(flagged_pairs, per_image_rows):
    """Choose best, typical, and hardest cases for the Median 3x3 filter."""
    median_rows = [
        row for row in per_image_rows
        if row["filter_key"] == "median_3"
    ]
    gains = []
    baseline_lookup = {
        row["image_id"]: row["ssim"]
        for row in per_image_rows
        if row["filter_key"] == "no_filter"
    }

    for row in median_rows:
        gains.append((
            row["ssim"] - baseline_lookup[row["image_id"]],
            row["image_id"],
        ))

    gains.sort()
    candidates = [
        gains[0][1],
        gains[len(gains) // 2][1],
        gains[-1][1],
    ]

    # Keep the examples distinct in case two cases have equal SSIM gains.
    chosen = []
    for image_id in candidates:
        if image_id not in chosen:
            chosen.append(image_id)

    return chosen


def save_before_after_examples(flagged_pairs, per_image_rows):
    """Create a visual input/filter/reference comparison for selected IDs."""
    pair_by_id = {
        pair["image_id"]: pair for pair in flagged_pairs
    }
    example_ids = choose_example_ids(flagged_pairs, per_image_rows)

    metric_lookup = {
        (row["image_id"], row["filter_key"]): row
        for row in per_image_rows
    }

    fig, axes = plt.subplots(
        len(example_ids),
        3,
        figsize=(15, 4.5 * len(example_ids)),
        squeeze=False,
    )

    for row_index, image_id in enumerate(example_ids):
        pair = pair_by_id[image_id]
        original = pair["noisy"]
        filtered = apply_filter(original, "median_3")
        clean = pair["clean"]

        original_metrics = metric_lookup[(image_id, "no_filter")]
        filtered_metrics = metric_lookup[(image_id, "median_3")]

        panels = [
            ("QSD1-W3 input", original),
            (
                "Median 3x3\n"
                f"PSNR {filtered_metrics['psnr_db']:.2f} dB | "
                f"SSIM {filtered_metrics['ssim']:.3f}",
                filtered,
            ),
            (
                "Clean reference\n"
                f"Input PSNR {original_metrics['psnr_db']:.2f} dB | "
                f"SSIM {original_metrics['ssim']:.3f}",
                clean,
            ),
        ]

        for column_index, (title, image) in enumerate(panels):
            axes[row_index, column_index].imshow(to_rgb(image))
            axes[row_index, column_index].set_title(
                f"Image ID {image_id}: {title}"
            )
            axes[row_index, column_index].axis("off")

    fig.suptitle(
        "Visual examples: input, selected filter, and clean reference",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_before_after_examples.png", dpi=180)
    plt.close(fig)

    return example_ids


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main():
    """Run the full denoising comparison and save all outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_matched_pairs()
    detections = detect_noisy_pairs(pairs)
    flagged_pairs, per_image_rows = evaluate_filters(pairs, detections)
    summary = summarize_metrics(per_image_rows)

    save_csv_outputs(detections, per_image_rows, summary)
    save_slide_notes(detections, summary)

    save_detection_chart(detections)
    save_filter_scorecard(summary)
    save_ssim_gain_heatmap(flagged_pairs, per_image_rows)
    save_metric_tradeoff(summary)
    example_ids = save_before_after_examples(
        flagged_pairs,
        per_image_rows,
    )

    impulse_ids = [
        row["image_id"] for row in detections if row["impulse_flag"]
    ]
    additive_ids = [
        row["image_id"] for row in detections if row["additive_flag"]
    ]
    flagged_ids = [
        row["image_id"] for row in detections if row["noisy_flag"]
    ]

    best_psnr = max(summary, key=lambda row: row["mean_psnr_db"])
    best_ssim = max(summary, key=lambda row: row["mean_ssim"])

    print(f"Matched QSD1-W3 images: {len(pairs)}")
    print(f"Impulse detector IDs: {impulse_ids}")
    print(f"Additive detector IDs: {additive_ids}")
    print(f"Flagged by either detector: {flagged_ids}")
    print(f"Images evaluated per filter: {len(flagged_pairs)}")
    print(
        f"Best mean PSNR: {best_psnr['method']} "
        f"({best_psnr['mean_psnr_db']:.2f} dB)"
    )
    print(
        f"Best mean SSIM: {best_ssim['method']} "
        f"({best_ssim['mean_ssim']:.3f})"
    )
    print(f"Visual example IDs: {example_ids}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()