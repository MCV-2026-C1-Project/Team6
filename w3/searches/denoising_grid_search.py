"""Compare denoising filters by their effect on QSD1 retrieval.

The script detects impulse-like noise in QSD1 queries, filters only flagged
queries, and evaluates retrieval against QSD1 ground truth. BBDD images remain
unchanged for every comparison.

Outputs are written to w3/results/denoising/.
"""

from pathlib import Path
import csv
import math
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from w2.utils import (  # noqa: E402
    color_light_block_descriptors,
    get_distances,
    mapk,
    retrieve,
)


BBDD_DIR = ROOT_DIR / "data" / "BBDD"
QSD1_DIR = ROOT_DIR / "data" / "qsd1_w3"
RESULTS_DIR = ROOT_DIR / "w3" / "results" / "denoising"

METRIC = "histogram_intersection"
DESCRIPTOR_PARAMS = {"grid": 4, "ab_bins": 48, "l_bins": 16}
NOISE_THRESHOLD = 0.01
K = 5

# Ordered from simpler filters to more complex ones. This order is used only
# to break ties when retrieval scores are identical.
FILTERS = [
    ("mean_5", "Mean 5x5"),
    ("gaussian_3", "Gaussian 3x3"),
    ("median_3", "Median 3x3"),
    ("bilateral_5", "Bilateral d=5"),
    ("bilateral_9", "Bilateral d=9"),
    ("nlm_5", "Non-local means h=5"),
    ("nlm_10", "Non-local means h=10"),
    ("nlm_15", "Non-local means h=15"),
]


def load_images(folder):
    """Load JPG images and their numeric IDs from a dataset folder."""
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images found in {folder}")

    ids = []
    images = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")

        ids.append(int(path.stem.split("_")[-1]))
        images.append(image)

    return ids, images


def impulse_noise_fraction(image_bgr):
    """Estimate the fraction of isolated grayscale pixels in an image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    height, width = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    isolated = np.ones((height, width), dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue

            neighbor = padded[
                1 + dy : 1 + dy + height,
                1 + dx : 1 + dx + width,
            ]
            isolated &= np.abs(gray - neighbor) > 30

    return float(isolated.mean())


def apply_candidate_filter(image_bgr, filter_name):
    """Apply one candidate filter used in this experiment."""
    if filter_name == "mean_5":
        return cv2.blur(image_bgr, (5, 5))

    if filter_name == "gaussian_3":
        return cv2.GaussianBlur(image_bgr, (3, 3), 0)

    if filter_name == "median_3":
        return cv2.medianBlur(image_bgr, 3)

    if filter_name == "bilateral_5":
        return cv2.bilateralFilter(image_bgr, 5, 75, 75)

    if filter_name == "bilateral_9":
        return cv2.bilateralFilter(image_bgr, 9, 75, 75)

    if filter_name.startswith("nlm_"):
        strength = int(filter_name.split("_")[1])
        return cv2.fastNlMeansDenoisingColored(
            image_bgr,
            None,
            strength,
            strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    raise ValueError(f"Unknown filter: {filter_name}")


def make_filtered_queries(query_images, noisy_indices, filter_name=None):
    """Filter flagged queries, or return the original queries for the baseline."""
    processed = list(query_images)

    if filter_name is None:
        return processed

    for index in noisy_indices:
        processed[index] = apply_candidate_filter(
            query_images[index],
            filter_name,
        )

    return processed


def compute_distances_in_batches(query_descriptors, db_descriptors):
    """Compute distances in small batches to limit temporary memory use."""
    batches = []
    batch_size = 2

    for start in range(0, len(query_descriptors), batch_size):
        batches.append(
            get_distances(
                query_descriptors[start : start + batch_size],
                db_descriptors,
                metric=METRIC,
            )
        )

    return np.vstack(batches)


def evaluate_retrieval(query_ids, query_images, db_ids, db_descriptors, ground_truth):
    """Return mAP@1, mAP@5, and ranked database IDs for these queries."""
    query_descriptors = color_light_block_descriptors(
        query_images,
        **DESCRIPTOR_PARAMS,
    )

    distances = compute_distances_in_batches(query_descriptors, db_descriptors)
    ranked_ids = retrieve(distances, db_ids, k=K)
    actual = [ground_truth[query_id] for query_id in query_ids]

    return (
        float(mapk(actual, ranked_ids, 1)),
        float(mapk(actual, ranked_ids, K)),
        ranked_ids,
    )


def save_results_csv(rows):
    """Save the baseline and filter scores to a CSV file."""
    path = RESULTS_DIR / "filter_results.csv"
    fields = [
        "filter",
        "mAP@1",
        "mAP@5",
        "flagged_query_mAP@5",
        "change_vs_baseline_mAP@5",
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        baseline_map5 = rows[0]["mAP@5"]
        for row in rows:
            writer.writerow({
                "filter": row["label"],
                "mAP@1": f"{row['mAP@1']:.4f}",
                "mAP@5": f"{row['mAP@5']:.4f}",
                "flagged_query_mAP@5": (
                    f"{row['flagged_query_mAP@5']:.4f}"
                    if row["flagged_query_mAP@5"] is not None else ""
                ),
                "change_vs_baseline_mAP@5": (
                    f"{row['mAP@5'] - baseline_map5:+.4f}"
                ),
            })


def save_detection_csv(query_ids, noise_fractions):
    """Save each query's detector score and whether it passed the threshold."""
    path = RESULTS_DIR / "noise_detection.csv"

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query_id", "impulse_fraction", "flagged"],
        )
        writer.writeheader()

        for query_id, fraction in zip(query_ids, noise_fractions):
            writer.writerow({
                "query_id": query_id,
                "impulse_fraction": f"{fraction:.6f}",
                "flagged": fraction > NOISE_THRESHOLD,
            })


def save_map_chart(rows, selected_label):
    """Save a ranked mAP@5 chart with the unfiltered baseline marked."""
    ranked = sorted(rows, key=lambda row: row["mAP@5"])
    labels = [row["label"] for row in ranked]
    scores = [row["mAP@5"] for row in ranked]
    baseline_score = rows[0]["mAP@5"]

    colors = []
    for row in ranked:
        if row["label"] == selected_label:
            colors.append("#E6863B")
        elif row["label"] == "No filter":
            colors.append("#777777")
        else:
            colors.append("#6B9AC4")

    figure_height = max(4.5, 0.42 * len(ranked))
    fig, axis = plt.subplots(figsize=(9, figure_height))
    bars = axis.barh(labels, scores, color=colors)

    axis.axvline(
        baseline_score,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"No-filter baseline: {baseline_score:.3f}",
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("mAP@5 across all QSD1 queries (higher is better)")
    axis.set_title("Retrieval score by denoising filter")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right", frameon=False)

    for bar, score in zip(bars, scores):
        axis.text(
            min(score + 0.015, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "map_at_5_comparison.png", dpi=160)
    plt.close(fig)


def save_visual_grid(query_id, image_bgr):
    """Save a side-by-side comparison for one detected noisy query."""
    panels = [("Original", image_bgr)]

    for filter_name, label in FILTERS:
        panels.append((
            label,
            apply_candidate_filter(image_bgr, filter_name),
        ))

    columns = 3
    row_count = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(12, 4 * row_count),
    )
    axes = np.asarray(axes).reshape(-1)

    for axis, (title, image) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")

    for axis in axes[len(panels):]:
        axis.axis("off")

    fig.suptitle(f"Query {query_id}: original and filter examples")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "filter_visual_comparison.png", dpi=160)
    plt.close(fig)


def save_slide_decision(rows, selected, noisy_ids, query_count):
    """Write a slide-ready explanation using the results from this run."""
    baseline = rows[0]
    delta_map5 = selected["mAP@5"] - baseline["mAP@5"]
    delta_map1 = selected["mAP@1"] - baseline["mAP@1"]

    tied = [
        row for row in rows[1:]
        if np.isclose(row["mAP@5"], selected["mAP@5"])
        and np.isclose(row["mAP@1"], selected["mAP@1"])
    ]

    lines = [
        "# Week 3 denoising: decision for slides",
        "",
        "## Decision",
        "",
    ]

    if selected["label"] == "No filter":
        lines += [
            "**Decision: do not apply a denoising filter.**",
            "",
            "No candidate improved all-query mAP@5 over the unfiltered baseline.",
        ]
    else:
        lines += [
            f"**Selected filter: {selected['label']}.**",
            "",
            f"On this QSD1 run, it achieved mAP@1 = "
            f"**{selected['mAP@1']:.4f}** and mAP@5 = "
            f"**{selected['mAP@5']:.4f}**.",
            "",
            f"The unfiltered baseline was mAP@1 = {baseline['mAP@1']:.4f} "
            f"and mAP@5 = {baseline['mAP@5']:.4f}. "
            f"The selected setup changed the scores by "
            f"{delta_map1:+.4f} mAP@1 and {delta_map5:+.4f} mAP@5.",
        ]

        if len(tied) > 1:
            tied_labels = ", ".join(row["label"] for row in tied)
            lines += [
                "",
                f"Several filters tied on both retrieval scores: {tied_labels}. "
                f"The simpler filter, {selected['label']}, was selected as the "
                "tie-break.",
            ]

    lines += [
        "",
        "## Test method",
        "",
        f"- Dataset: {query_count} QSD1 development queries with correspondence "
        "ground truth.",
        f"- Noise detector: flag a query when more than {NOISE_THRESHOLD:.0%} "
        "of grayscale pixels differ by more than 30 intensity levels from "
        "all eight neighbors.",
        f"- Flagged query IDs in this run: {noisy_ids}.",
        "- Each candidate filter was applied only to flagged queries. "
        "Unflagged queries and BBDD images were left unchanged.",
        "- The Week 2 color-light descriptor and histogram-intersection "
        "distance were kept fixed. The comparison metric was retrieval mAP.",
        "",
        "## Limitation",
        "",
        "This experiment measures retrieval performance; it does not calculate "
        "PSNR or SSIM. Those pixelwise measures require paired clean reference "
        "images. The comparison uses the Week 2 descriptor, so repeat the "
        "evaluation with the final Week 3 descriptor before reporting the "
        "team's final result.",
        "",
        "## Short presentation wording",
        "",
    ]

    if selected["label"] == "No filter":
        lines.append(
            "“We compared the unfiltered baseline with several denoising "
            "filters on the detected noisy development queries. Filtering did "
            "not improve mAP@5, so we kept the images unchanged.”"
        )
    else:
        lines.append(
            f"“We compared the unfiltered baseline with eight filters on the "
            f"{len(noisy_ids)} detected noisy development queries. "
            f"{selected['label']} gave the best retrieval result "
            f"(mAP@5 {selected['mAP@5']:.4f} versus "
            f"{baseline['mAP@5']:.4f} without filtering). We selected it "
            "because it improved retrieval and, where scores tied, it was "
            "the simpler option.”"
        )

    path = RESULTS_DIR / "decision_for_slides.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QSD1_DIR)

    gt_path = QSD1_DIR / "gt_corresps.pkl"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    with open(gt_path, "rb") as file:
        gt_list = pickle.load(file)

    if len(query_ids) != len(gt_list):
        raise ValueError(
            f"Found {len(query_ids)} queries but {len(gt_list)} "
            "ground-truth entries."
        )

    ground_truth = {query_id: gt_list[query_id] for query_id in query_ids}

    noise_fractions = [
        impulse_noise_fraction(image)
        for image in query_images
    ]
    noisy_indices = [
        index for index, fraction in enumerate(noise_fractions)
        if fraction > NOISE_THRESHOLD
    ]
    noisy_ids = [query_ids[index] for index in noisy_indices]

    print(f"QSD1 queries: {len(query_ids)}")
    print(f"Flagged noisy queries: {noisy_ids}")
    print(f"Noise threshold: {NOISE_THRESHOLD:.1%}")

    save_detection_csv(query_ids, noise_fractions)

    # The database is described once and stays identical in every comparison.
    db_descriptors = color_light_block_descriptors(
        db_images,
        **DESCRIPTOR_PARAMS,
    )

    # Evaluate the unfiltered baseline.
    map1, map5, _ = evaluate_retrieval(
        query_ids,
        query_images,
        db_ids,
        db_descriptors,
        ground_truth,
    )
    rows = [{
        "filter": "no_filter",
        "label": "No filter",
        "mAP@1": map1,
        "mAP@5": map5,
        "flagged_query_mAP@5": None,
        "order": -1,
    }]
    print(f"{'No filter':20s} mAP@1={map1:.4f}  mAP@5={map5:.4f}")

    # Evaluate each candidate with the same detection and retrieval setup.
    for order, (filter_name, label) in enumerate(FILTERS):
        processed_queries = make_filtered_queries(
            query_images,
            noisy_indices,
            filter_name,
        )

        map1, map5, _ = evaluate_retrieval(
            query_ids,
            processed_queries,
            db_ids,
            db_descriptors,
            ground_truth,
        )

        flagged_map5 = None
        if noisy_indices:
            flagged_ids = [query_ids[index] for index in noisy_indices]
            flagged_images = [
                processed_queries[index] for index in noisy_indices
            ]
            _, flagged_map5, _ = evaluate_retrieval(
                flagged_ids,
                flagged_images,
                db_ids,
                db_descriptors,
                ground_truth,
            )

        rows.append({
            "filter": filter_name,
            "label": label,
            "mAP@1": map1,
            "mAP@5": map5,
            "flagged_query_mAP@5": flagged_map5,
            "order": order,
        })

        output = (
            f"{label:20s} mAP@1={map1:.4f}  mAP@5={map5:.4f}"
        )
        if flagged_map5 is not None:
            output += f"  flagged mAP@5={flagged_map5:.4f}"
        print(output)

    # Select by all-query mAP@5, then mAP@1, then simpler filter order.
    best_candidate = max(
        rows[1:],
        key=lambda row: (
            row["mAP@5"],
            row["mAP@1"],
            -row["order"],
        ),
    )

    # Do not select a filter if it fails to beat the unfiltered baseline.
    if best_candidate["mAP@5"] > rows[0]["mAP@5"]:
        selected = best_candidate
    else:
        selected = rows[0]

    save_results_csv(rows)
    save_map_chart(rows, selected["label"])
    save_slide_decision(rows, selected, noisy_ids, len(query_ids))

    if noisy_indices:
        first_index = noisy_indices[0]
        save_visual_grid(query_ids[first_index], query_images[first_index])

    print(f"\nSelected setup: {selected['label']}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()