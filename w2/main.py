from utils import *
import pickle
import pandas as pd


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w1"

METRIC = "histogram_intersection"


if __name__ == "__main__":
    # Load database, query images, and ground-truth correspondences.
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QSD1)

    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    # Compute descriptors and distances for both methods.
    db_m1 = simple_descriptors(db_images)
    query_m1 = simple_descriptors(query_images)
    db_m2 = complex_descriptors(db_images)
    query_m2 = complex_descriptors(query_images)

    dists_m1 = get_distances(query_m1, db_m1, metric=METRIC)
    dists_m2 = get_distances(query_m2, db_m2, metric=METRIC)

    # Retrieve the top-5 results and compute mAP@1 and mAP@5.
    topk5_m1 = retrieve(dists_m1, db_ids, k=5)
    m1_mapk1 = mapk(ground_truth, topk5_m1, k=1)
    m1_mapk5 = mapk(ground_truth, topk5_m1, k=5)

    topk5_m2 = retrieve(dists_m2, db_ids, k=5)
    m2_mapk1 = mapk(ground_truth, topk5_m2, k=1)
    m2_mapk5 = mapk(ground_truth, topk5_m2, k=5)

    rows = [
        ("Method 1", m1_mapk1, m1_mapk5),
        ("Method 2", m2_mapk1, m2_mapk5),
    ]

    print("\nResults:")
    sep = "-" * 32
    print(sep)
    print(f"{'':<10}{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)

    for name, k1, k5 in rows:
        print(f"{name:<10}{k1:>10.4f}{k5:>10.4f}")

    print(sep)