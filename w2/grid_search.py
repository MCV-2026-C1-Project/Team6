import csv
import importlib.util
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from utils import spatial_descriptor_3d


ROOT = Path(__file__).resolve().parent.parent

# Reuse the team's existing retrieval and evaluation functions.
spec = importlib.util.spec_from_file_location(
    "week1_utils", ROOT / "w1" / "utils.py"
)
week1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(week1)


def build_configurations() -> list[dict]:
    """
    Define focused experiments around the strongest first-round descriptors.

    Args:
        None.

    Returns:
        List of 20 configurations specifying bins, grids and level weights.
    """
    bin_options = [
        (2, 8, 8),
        (3, 8, 8),
        (4, 6, 6),
        (4, 8, 8),
        (4, 12, 12),
        (6, 8, 8),
        (8, 8, 8),
    ]

    configurations = []

    # Fourteen block configurations.
    for grid in (3, 4):
        for bins in bin_options:
            configurations.append({
                "name": f"blocks_{grid}_bins_{'_'.join(map(str, bins))}",
                "bins": bins,
                "grids": (grid,),
                "weights": (1.0,),
            })

    # Six pyramid configurations: equal and fine-emphasized weights.
    for bins in ((4, 8, 8), (4, 12, 12)):
        for weights in (
            (1.0, 1.0, 1.0),
            (1.0, 2.0, 4.0),
            (1.0, 1.0, 4.0),
        ):
            configurations.append({
                "name": (
                    f"pyramid_1_2_4_bins_{'_'.join(map(str, bins))}"
                    f"_weights_{'_'.join(str(int(w)) for w in weights)}"
                ),
                "bins": bins,
                "grids": (1, 2, 4),
                "weights": weights,
            })

    return configurations


def compare_descriptors(
    query_desc: np.ndarray,
    db_desc: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Compare descriptors using the team's distance functions in small batches.

    Args:
        query_desc: Matrix with one descriptor per query.
        db_desc: Matrix with one descriptor per museum image.
        metric: Name of a comparison measure supported by Week 1 utilities.

    Returns:
        Distance matrix of shape (n_queries, n_database).
        Lower values indicate better matches.
    """
    distances = np.empty(
        (len(query_desc), len(db_desc)), dtype=np.float32
    )

    for q in range(len(query_desc)):
        for first in range(0, len(db_desc), 32):
            last = min(first + 32, len(db_desc))
            distances[q:q + 1, first:last] = week1.get_distances(
                query_desc[q:q + 1],
                db_desc[first:last],
                metric=metric,
            )

    if not np.all(np.isfinite(distances)):
        raise ValueError(f"Non-finite distances produced by {metric}.")

    return distances


if __name__ == "__main__":
    db_ids, db_images = week1.load_images(ROOT / "data" / "BBDD")
    query_ids, query_images = week1.load_images(
        ROOT / "data" / "qsd1_w1"
    )

    with open(
        ROOT / "data" / "qsd1_w1" / "gt_corresps.pkl", "rb"
    ) as file:
        ground_truth = pickle.load(file)

    if not ground_truth or query_ids != list(range(len(ground_truth))):
        raise ValueError(
            "Queries do not match the complete ground truth. "
            "Check for missing images."
        )

    if len(db_ids) < 5 or len(set(db_ids)) != len(db_ids):
        raise ValueError("Need at least five unique museum image IDs.")

    db_id_set = set(db_ids)
    if any(
        int(correct_id) not in db_id_set
        for matches in ground_truth
        for correct_id in matches
    ):
        raise ValueError("A ground-truth painting is missing from BBDD.")

    output = ROOT / "w2" / "results"
    output.mkdir(parents=True, exist_ok=True)

    metrics = ["histogram_intersection", "hellinger", "chi2"]
    configurations = build_configurations()

    # Recompute the baseline with the same images and evaluation code.
    print("Computing Week 1 Method 2 baseline...", flush=True)
    baseline_db = week1.complex_descriptors(db_images)
    baseline_query = week1.complex_descriptors(query_images)

    baseline_distances = compare_descriptors(
        baseline_query, baseline_db, "histogram_intersection"
    )
    baseline_predictions = week1.retrieve(
        baseline_distances, db_ids, k=5
    )

    baseline_map1 = float(
        week1.mapk(ground_truth, baseline_predictions, k=1)
    )
    baseline_map5 = float(
        week1.mapk(ground_truth, baseline_predictions, k=5)
    )

    print(
        f"Baseline: mAP@1={baseline_map1:.4f}, "
        f"mAP@5={baseline_map5:.4f}",
        flush=True,
    )

    baseline_top1 = [
        prediction[0] in correct
        for prediction, correct in zip(
            baseline_predictions, ground_truth
        )
    ]

    details = {
        "baseline": {
            "mAP@1": baseline_map1,
            "mAP@5": baseline_map5,
            "query_ids": [int(q) for q in query_ids],
            "top5": baseline_predictions,
        },
        "experiments": {},
    }
    results = []

    del baseline_db, baseline_query, baseline_distances

    for index, config in enumerate(configurations, start=1):
        print(
            f"\n[{index}/{len(configurations)}] {config['name']}",
            flush=True,
        )

        start = perf_counter()

        db_desc = np.stack([
            spatial_descriptor_3d(
                image,
                bins=config["bins"],
                grids=config["grids"],
                weights=config["weights"],
            )
            for image in db_images
        ])

        query_desc = np.stack([
            spatial_descriptor_3d(
                image,
                bins=config["bins"],
                grids=config["grids"],
                weights=config["weights"],
            )
            for image in query_images
        ])

        extraction_seconds = perf_counter() - start

        # Check the descriptor contract before comparing images.
        for descriptors in (db_desc, query_desc):
            if (
                not np.all(np.isfinite(descriptors))
                or np.any(descriptors < 0)
                or not np.allclose(
                    descriptors.sum(axis=1), 1.0, atol=1e-5
                )
            ):
                raise ValueError("Invalid or unnormalized descriptors.")

        # Extract once, then reuse descriptors for all three measures.
        for metric in metrics:
            start = perf_counter()
            distances = compare_descriptors(
                query_desc, db_desc, metric
            )
            predictions = week1.retrieve(distances, db_ids, k=5)
            matching_seconds = perf_counter() - start

            map1 = float(
                week1.mapk(ground_truth, predictions, k=1)
            )
            map5 = float(
                week1.mapk(ground_truth, predictions, k=5)
            )

            query_details = []
            rescued = []
            lost = []

            for i, query_id in enumerate(query_ids):
                correct = ground_truth[i]
                top1_correct = predictions[i][0] in correct

                if top1_correct and not baseline_top1[i]:
                    rescued.append(int(query_id))
                if baseline_top1[i] and not top1_correct:
                    lost.append(int(query_id))

                query_details.append({
                    "query_id": int(query_id),
                    "correct_ids": [int(value) for value in correct],
                    "top5": [int(value) for value in predictions[i]],
                    "top1_correct": bool(top1_correct),
                    "baseline_top1_correct": bool(baseline_top1[i]),
                    "first_correct_rank_in_top5": next(
                        (
                            rank
                            for rank, value in enumerate(
                                predictions[i], start=1
                            )
                            if value in correct
                        ),
                        None,
                    ),
                })

            experiment_name = f"{config['name']}__{metric}"

            results.append({
                "configuration": config["name"],
                "metric": metric,
                "bins": str(config["bins"]),
                "grids": str(config["grids"]),
                "weights": str(config["weights"]),
                "dimensions": int(db_desc.shape[1]),
                "mAP@1": map1,
                "mAP@5": map5,
                "delta_vs_week1": map5 - baseline_map5,
                "top1_correct_count": sum(
                    item["top1_correct"] for item in query_details
                ),
                "rescued_count": len(rescued),
                "lost_count": len(lost),
                "extraction_seconds": extraction_seconds,
                "matching_seconds": matching_seconds,
                "bytes_per_descriptor": int(db_desc[0].nbytes),
                "query_nonzero_fraction": float(
                    np.count_nonzero(query_desc) / query_desc.size
                ),
            })

            details["experiments"][experiment_name] = {
                "settings": config,
                "metric": metric,
                "rescued_query_ids": rescued,
                "lost_query_ids": lost,
                "queries": query_details,
            }

            # Rank by accuracy; prefer smaller descriptors on exact ties.
            ranked = sorted(
                results,
                key=lambda row: (
                    -row["mAP@5"],
                    -row["mAP@1"],
                    row["dimensions"],
                ),
            )

            with open(
                output / "3d_refined_results.csv", "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(ranked[0])
                )
                writer.writeheader()
                writer.writerows(ranked)

            with open(
                output / "3d_refined_query_results.json", "w"
            ) as file:
                json.dump(details, file, indent=2)

            print(
                f"  {metric:24s} "
                f"mAP@1={map1:.4f} | mAP@5={map5:.4f} | "
                f"rescued={len(rescued)} | lost={len(lost)}",
                flush=True,
            )

    print("\nTOP 10 EXPERIMENTS")
    for row in ranked[:10]:
        print(
            f"{row['configuration']} | {row['metric']} | "
            f"mAP@5={row['mAP@5']:.4f} | "
            f"length={row['dimensions']}"
        )