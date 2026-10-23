from utils import *
import pickle
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd2_w3"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GROUND_TRUTH_DIR = QUERY_DIR / "gt_corresps.pkl"

RUN_TESTS = True
TEST1_DIR = DATA_DIR / "qst1_w3"
TEST2_DIR = DATA_DIR / "qst2_w3"
K = 10

DESC_PARAMS = {
    "descriptor": dct_pyramid_descriptor,
    "color_code": cv2.COLOR_BGR2YCrCb,
    "channels": [0, 1, 2],
    "size": 672,
    "metric": "canberra",
    "kwargs": {
        "levels": [1,2,3],
        "n_coef": 32,
        "skip_dc": False,
    },
}

def process_queries(query_images: list[np.ndarray], n_feat: int, params: dict):
    """
    Detect up to MAX_PAINTINGS paintings per query and describe each of them.

    Returns:
        descs: (n_queries, MAX_PAINTINGS, n_feat) array, NaN rows where a
            painting does not exist (all NaN if nothing is detected).
        masks_by_query: detected masks of each query.
    """
    descs = np.full((len(query_images), MAX_PAINTINGS, n_feat), np.nan, np.float32)
    masks_by_query = []

    for i, img in enumerate(tqdm(query_images, desc="Queries")):
        img = denoise_image(img)
        masks = detect_paintings(img)
        masks_by_query.append(masks)

        for j, mask in enumerate(masks[:MAX_PAINTINGS]):
            descs[i, j] = describe(crop_to_mask(img, mask), params)

    return descs, masks_by_query


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    with open(GROUND_TRUTH_DIR, "rb") as f:
        ground_truth = pickle.load(f)

    ground_truth = [ground_truth[qid] for qid in query_ids]

    print("Computing BBDD descriptors...")
    db_descs = np.vstack([describe(img, DESC_PARAMS) for img in tqdm(db_images, desc="BBDD")])

    print("Detecting paintings and computing query descriptors...")
    query_descs, masks_by_query = process_queries(query_images, db_descs.shape[1], DESC_PARAMS)
    print(f"Query descriptors: {query_descs.shape}")

    # (n_queries, MAX_PAINTINGS, n_database), NaN rows for missing paintings
    dists = get_distances(query_descs, db_descs, metric=DESC_PARAMS["metric"])

    predictions = retrieve_multi(dists, db_ids, k=K)
    mapk1 = mapk_multi(ground_truth, predictions, k=1)
    mapk5 = mapk_multi(ground_truth, predictions, k=5)

    n_detected = [len(m) for m in masks_by_query]
    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"Metric: {DESC_PARAMS['metric']}")
    print(f"Detections: 0 -> {n_detected.count(0)}, 1 -> {n_detected.count(1)}, "
          f"2 -> {n_detected.count(2)}")
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{mapk1:>10.4f}{mapk5:>10.4f}")
    print(sep)

    if RUN_TESTS:
        tests = [
            ("QST1", TEST1_DIR, False),
            ("QST2", TEST2_DIR, True),
        ]

        for name, test_dir, save_masks in tests:
            print(f"\nRunning test {name}...")
            test_ids, test_images = load_images(test_dir)

            test_descs, test_masks = process_queries(test_images, db_descs.shape[1], DESC_PARAMS)
            test_dists = get_distances(test_descs, db_descs, metric=DESC_PARAMS["metric"])
            test_predictions = retrieve_multi(test_dists, db_ids, k=K)

            n_detected_test = [len(m) for m in test_masks]
            print(f"{name} detections: 0 -> {n_detected_test.count(0)}, "
                  f"1 -> {n_detected_test.count(1)}, 2 -> {n_detected_test.count(2)}")

            out_test = RESULTS_DIR / name / "method1"
            out_test.mkdir(parents=True, exist_ok=True)
            with open(out_test / "result.pkl", "wb") as f:
                pickle.dump(test_predictions, f)

            if save_masks:
                for i, (qid, img, masks) in enumerate(zip(test_ids, test_images, test_masks)):
                    stem = Path(str(qid)).stem
                    idx = int(stem) if stem.isdigit() else i
                    cv2.imwrite(str(out_test / f"{idx:05d}.png"), combined_mask(masks, img.shape))

            print(f"{name} results saved to {out_test}")