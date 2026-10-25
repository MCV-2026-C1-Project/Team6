from utils import *
import pickle
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from w2.utils import color_light_block_descriptors

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w3"

METRIC = "histogram_intersection"
DESC_PARAMS = {"grid": 6, "ab_bins": 24, "l_bins": 4}

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

    print("Computing descriptors...")
    db_desc = color_light_block_descriptors(db_images, masks=None, **DESC_PARAMS)
    q_desc = color_light_block_descriptors(query_images, masks=None, **DESC_PARAMS)

    dists = get_distances(q_desc, db_desc, metric=METRIC)

    # Retrieve the top-5 results and compute mAP@1 and mAP@5.
    results = dict(zip(query_ids, retrieve(dists, db_ids, k=5)))
    map1 = evaluate(ground_truth, results, k=1)
    map5 = evaluate(ground_truth, results, k=5)

    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"Shift: {'on, max ' + str(MAX_SHIFT * 100) + '%' if APPLY_SHIFT else 'off'}")
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{map1:>10.4f}{map5:>10.4f}")
    print(sep)