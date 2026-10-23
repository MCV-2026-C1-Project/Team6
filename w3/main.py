from utils import *
import pickle
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd2_w3"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

GROUND_TRUTH_DIR = QUERY_DIR / "gt_corresps.pkl"

SAVE_VISUALIZATIONS = True
SAVE_RESULTS = True
K = 10

# PARAMS = {
#     "descriptor": dct_block_descriptor,
#     "color_code": cv2.COLOR_BGR2YCrCb,
#     "channels": [0, 1, 2],
#     "size": 672,
#     "metric": "cityblock",
#     "kwargs": {
#         "n_blocks": 6,
#         "n_coef": 20,
#         "skip_dc": True,
#     },
# }

PARAMS = {
    "descriptor": dct_pyramid_descriptor,
    "color_code": cv2.COLOR_BGR2YCrCb,
    "channels": [0, 1, 2],
    "size": 672,
    "metric": "canberra",
    "kwargs": {
        "n_coef": 32,
        "skip_dc": False,
    },
}


# =============================================================================
# HELPERS
# =============================================================================


def describe_queries(query_images: list[np.ndarray], n_feat: int, params: dict):
    """
    Detect up to MAX_PAINTINGS paintings per query and describe each of them.

    Returns:
        descs: (n_queries, MAX_PAINTINGS, n_feat) array, NaN rows where a
            painting does not exist.
        masks_by_query: detected masks of each query (for visualization).
    """
    descs = np.full((len(query_images), MAX_PAINTINGS, n_feat), np.nan, np.float32)
    masks_by_query = []

    for i, img in enumerate(tqdm(query_images, desc="Queries")):
        img = denoise_image(img)
        masks = detect_paintings(img)
        masks_by_query.append(masks)

        # If nothing is detected, fall back to the whole image as one painting
        paintings = [crop_to_mask(img, m) for m in masks] or [img]
        for j, painting in enumerate(paintings[:MAX_PAINTINGS]):
            descs[i, j] = describe(painting, params)

    return descs, masks_by_query


def retrieve_multi(dists: np.ndarray, db_ids: list[int], k: int) -> list[list[list[int]]]:
    """
    Top-k retrieval for a (n_queries, MAX_PAINTINGS, n_database) distance array.

    Returns:
        One list per query with one top-k list per existing painting:
        [[[ids painting 1], [ids painting 2]], [[ids painting 1]], ...]
    """
    preds = []
    for q in dists:
        valid = ~np.isnan(q).all(axis=1)
        preds.append(retrieve(q[valid], db_ids, k) if valid.any() else [])
    return preds


def mapk_multi(actual: list, predicted: list[list[list[int]]], k: int) -> float:
    """
    mAP@k computed per painting.

    actual[q] is the ground truth of query q, e.g. [120] or [120, 170]
    (one BBDD id per painting, in the same order as the detections).
    A painting in the ground truth that was not detected gets AP = 0.
    """
    aps = []
    for gt_q, pred_q in zip(actual, predicted):
        for j, gt_painting in enumerate(gt_q):
            gt_painting = list(gt_painting) if isinstance(gt_painting, (list, tuple)) else [gt_painting]
            pred = pred_q[j] if j < len(pred_q) else []
            aps.append(apk(gt_painting, pred, k))
    return float(np.mean(aps)) if aps else 0.0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    with open(GROUND_TRUTH_DIR, "rb") as f:
        ground_truth = pickle.load(f)

    ground_truth = [ground_truth[qid] for qid in query_ids]

    print("Computing BBDD descriptors...")
    db_descs = np.vstack([describe(img, PARAMS) for img in tqdm(db_images, desc="BBDD")])

    print("Detecting paintings and computing query descriptors...")
    query_descs, masks_by_query = describe_queries(query_images, db_descs.shape[1], PARAMS)
    print(f"Query descriptors: {query_descs.shape}")  # (n_queries, 2, n_features)

    # (n_queries, MAX_PAINTINGS, n_database), NaN rows for missing paintings
    dists = get_distances(query_descs, db_descs, metric=PARAMS["metric"])

    predictions = retrieve_multi(dists, db_ids, k=K)
    mapk1 = mapk_multi(ground_truth, predictions, k=1)
    mapk5 = mapk_multi(ground_truth, predictions, k=5)

    n_detected = [len(m) for m in masks_by_query]
    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"Metric: {PARAMS['metric']}")
    print(f"Detections: 0 -> {n_detected.count(0)}, 1 -> {n_detected.count(1)}, "
          f"2 -> {n_detected.count(2)}")
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{mapk1:>10.4f}{mapk5:>10.4f}")
    print(sep)

    if SAVE_VISUALIZATIONS:
        visualize_masks(query_images, masks_by_query, OUTPUT_DIR / "masks")
        visualize_painting_count(query_images, masks_by_query, OUTPUT_DIR)

    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "result.pkl", "wb") as f:
            pickle.dump(predictions, f)
        print(f"Predictions saved under: {OUTPUT_DIR / 'result.pkl'}")