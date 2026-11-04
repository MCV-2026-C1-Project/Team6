from utils import *
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from denoising import denoise_image
from descriptors import NORMS, describe
from segmentation import crop_to_mask


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w4"
GROUND_TRUTH_PATH = QUERY_DIR / "gt_corresps.pkl"
FRAMES_PATH = QUERY_DIR / "frames.pkl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

RUN_TESTS = True
TEST_DIR = DATA_DIR / "qst1_w4"
N_WORKERS = os.cpu_count()

PARAMS = {
    "descriptor": "rootsift",
    "detector": None,
    "color": "gray",
    "clahe": True,
    "max_keypoints": 1000,
    "max_side": 800,
    "lowe_ratio": 0.75,
    "ransac_reproj_px": 5.0,
    "min_inliers": 25,
    "k": 10,
}


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    query_images: list[np.ndarray],
    db_images: list[np.ndarray],
    db_ids: list[int],
    *,
    descriptor: str,
    detector: str | None,
    color: str,
    clahe: bool,
    max_keypoints: int,
    max_side: int,
    lowe_ratio: float,
    ransac_reproj_px: float,
    min_inliers: int,
    k: int,
    n_workers: int = N_WORKERS,
    predictions: bool = True,
) -> dict:
    """
    Full W4 pipeline: denoise -> crop paintings -> resize -> local features ->
    matching (Lowe's ratio + RANSAC) -> retrieval with unknowns (-1).

    Reusable from the grid search: with predictions=False it stops at the
    matching, and the returned "inliers" can be passed to retrieve_with_unknowns
    to sweep min_inliers and k without matching again.

    Args:
        query_images: Query images in BGR format.
        db_images: Database images in BGR format.
        db_ids: Database IDs.
        descriptor: Key of descriptors.DESCRIPTORS.
        detector: Key of keypoints.DETECTORS, or None for the descriptor's own detector.
        color: One of descriptors.COLORS.
        clahe: Equalise the local contrast before detecting and describing.
        max_keypoints: Maximum number of keypoints per image.
        max_side: Longest side of every painting before describing it.
        lowe_ratio: Max ratio of Lowe's ratio test.
        ransac_reproj_px: RANSAC reprojection threshold, in pixels.
        min_inliers: Inliers the best database image needs, below it -> -1.
        k: Number of IDs retrieved per painting.
        n_workers: Number of processes for the matching.
        predictions: Retrieve the top-k IDs of every painting.

    Returns:
        Dict with "inliers", "masks" and, if predictions is True, "predictions".
    """
    # ---------------------------------------------------------------- database
    db_features = []
    for image in tqdm(db_images, desc="BBDD"):
        painting = resize_max_side(image, max_side)
        db_features.append(describe(painting, descriptor, detector, color, clahe, max_keypoints))

    # ---------------------------------------------------------------- queries
    frames_by_query = load_pickle(FRAMES_PATH)
    query_features, query_masks = [], []

    for query, frames in zip(tqdm(query_images, desc="Queries"), frames_by_query):
        query = denoise_image(query)
        masks = [cv2.fillPoly(np.zeros(query.shape[:2], np.uint8), [np.int32(corners).reshape(-1, 1, 2)], 1)
                 for _angle, corners in frames]

        feats = []
        for mask in masks:
            cropped = crop_to_mask(query, mask)
            resized = resize_max_side(cropped, max_side)
            descr = describe(resized, descriptor, detector, color, clahe, max_keypoints)
            feats.append(descr)

        query_masks.append(masks)
        query_features.append(feats)

    # ---------------------------------------------------------------- matching
    inliers = inliers_by_query(query_features, db_features, NORMS[descriptor],
                               lowe_ratio, ransac_reproj_px, n_workers)

    results = {"inliers": inliers, "masks": query_masks}

    # ---------------------------------------------------------------- retrieval
    if predictions:
        results["predictions"] = retrieve_with_unknowns(inliers, db_ids, k, min_inliers)

    return results


# =============================================================================
# REPORT
# =============================================================================

def print_results(results: dict, ground_truth: list, params: dict):
    """
    Print the configuration of a run, its painting counts compared with the
    ground truth and its mAP.
    """
    predicted_counts = [len(masks) for masks in results["masks"]]
    ground_truth_counts = [len(paintings) for paintings in ground_truth]
    predicted_unknown = sum(painting == [UNKNOWN] for query in results["predictions"] for painting in query)
    ground_truth_unknown = sum(painting == UNKNOWN for query in ground_truth for painting in query)

    print("\nResults:")
    print(f"Descriptor: {params['descriptor']} ({params['detector'] or 'own'} keypoints)")

    separator = "-" * 52
    print(separator)
    print(f"{'':<14}{'0 paint.':>9}{'1 paint.':>9}{'2 paint.':>9}{'Unknown':>11}")
    print(separator)
    for name, counts, n_unknown in [("Predicted", predicted_counts, predicted_unknown),
                                    ("Ground truth", ground_truth_counts, ground_truth_unknown)]:
        print(f"{name:<14}{counts.count(0):>9}{counts.count(1):>9}{counts.count(2):>9}{n_unknown:>11}")
    print(separator)
    print(f"Total paintings (ground truth): {sum(ground_truth_counts)}")

    separator = "-" * 20
    print(separator)
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(separator)
    print(f"{results['map1']:>10.4f}{results['map5']:>10.4f}")
    print(separator)


def save_predictions(predictions: list, output_dir: Path):
    """Save the predictions as output_dir/result.pkl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    ground_truth_by_id = load_pickle(GROUND_TRUTH_PATH)
    ground_truth = [ground_truth_by_id[query_id] for query_id in query_ids]

    results = run_pipeline(query_images, db_images, db_ids, **PARAMS)

    results["map1"] = mapk_multi(ground_truth, results["predictions"], k=1)
    results["map5"] = mapk_multi(ground_truth, results["predictions"], k=5)
    print_results(results, ground_truth, PARAMS)

    if RUN_TESTS and TEST_DIR.exists():
        print("\nRunning test QST1...")
        _, test_images = load_images(TEST_DIR)
        test_results = run_pipeline(test_images, db_images, db_ids, **PARAMS)
        save_predictions(test_results["predictions"], RESULTS_DIR / "QST1" / "method1")
