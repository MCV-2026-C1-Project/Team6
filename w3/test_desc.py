from utils import *
import pickle
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w3"

PARAMS = {
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

APPLY_SHIFT = False
MAX_SHIFT = 0.05
SEED = 42


def random_shift(img, max_shift, rng):
    """Desplaça la imatge un nombre aleatori de píxels en x i y (màx. max_shift * mida)."""
    h, w = img.shape[:2]
    dx = int(rng.integers(-int(max_shift * w), int(max_shift * w) + 1))
    dy = int(rng.integers(-int(max_shift * h), int(max_shift * h) + 1))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    with open(QUERY_DIR / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

        rng = np.random.default_rng(SEED)
    if APPLY_SHIFT:
        query_images = [random_shift(im, MAX_SHIFT, rng) for im in query_images]

    print("Computing BBDD descriptors...")
    db_descs = np.vstack([describe(img, PARAMS) for img in tqdm(db_images, desc="BBDD")])

    print("Computing query descriptors...")
    query_descs = np.vstack([describe(img, PARAMS) for img in tqdm(query_images, desc="Queries")])

    dists = get_distances(query_descs, db_descs, metric=PARAMS["metric"])

    # Retrieve the top-5 results and compute mAP@1 and mAP@5.
    topk5 = retrieve(dists, db_ids, k=5)
    mapk1 = mapk(ground_truth, topk5, k=1)
    mapk5 = mapk(ground_truth, topk5, k=5)

    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"Shift: {'on, max ' + str(MAX_SHIFT * 100) + '%' if APPLY_SHIFT else 'off'}")
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{mapk1:>10.4f}{mapk5:>10.4f}")
    print(sep)