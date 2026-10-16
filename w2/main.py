from utils import *
import pickle
import pandas as pd


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w2"
QSD2 = DATA_DIR / "qsd2_w2"


METRIC = "histogram_intersection"

BEST_PARAMS = {
    "2d_descriptor": {
        "grid": 4,
        "ab_bins": 48,
        "l_bins": 16
    },
    "3d_descriptor": {
        "bins": (2, 8, 8),
        "grids": (4,),
    },
    "segmentation": {
        "prefilter": "none",
        "color_space": "HSV",
        "channels": [1, 2],
        "threshold": "otsu",
        "close_frac": 0.012306
    }
}

def task2(descriptor="2d_descriptor"):
    global BBDD_DIR, QSD1, METRIC, BEST_PARAMS

    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QSD1)
    params = BEST_PARAMS[descriptor]

    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    descriptor_function = {
        "2d_descriptor": color_light_block_descriptors,
        "3d_descriptor": spatial_descriptors_3d,
    }.get(descriptor)

    if descriptor_function is None:
        raise ValueError(f"Unknown descriptor: {descriptor}")

    db_desc = descriptor_function(db_images, **params)
    query_desc = descriptor_function(query_images, **params)

    dists = get_distances(query_desc, db_desc, metric=METRIC)
    topk5 = retrieve(dists, db_ids, k=5)
    results = dict(zip(query_ids, topk5))

    mapk1 = evaluate(ground_truth, results, k=1)
    mapk5 = evaluate(ground_truth, results, k=5)

    print(f"\nResults ({descriptor}, distance={METRIC}):")
    print("-" * 50)
    print(f"{'':<25}{'mAP@1':>10}{'mAP@5':>10}")
    print("-" * 50)
    print(f"{descriptor:<25}{mapk1:>10.4f}{mapk5:>10.4f}")
    print("-" * 50)

    print("\nFailed queries:")
    for qid, retrieved in zip(query_ids, topk5):
        expected = ground_truth[qid]
        if retrieved[0] not in expected:
            rank = next((i + 1 for i, r in enumerate(retrieved) if r in expected), None)
            print(f"  {qid}: expected {expected}, retrieved {retrieved} "
                  f"({'rank ' + str(rank) if rank else 'not in top-5'})")


if __name__ == "__main__":
    task2()