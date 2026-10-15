from utils import *
import pickle


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w2"

# Descriptor and distance configuration (best result of grid_search.py)
GRID = 4
AB_BINS = 48
L_BINS = 16
METRIC = "histogram_intersection"


if __name__ == "__main__":
    # Load database, query images, and ground-truth correspondences.
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QSD1)

    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    # Compute color + lightness block descriptors and distances.
    db_desc = color_light_block_descriptors(db_images, GRID, AB_BINS, L_BINS)
    query_desc = color_light_block_descriptors(query_images, GRID, AB_BINS, L_BINS)

    dists = get_distances(query_desc, db_desc, metric=METRIC)

    # Retrieve the top-5 results and compute mAP@1 and mAP@5.
    topk5 = retrieve(dists, db_ids, k=5)
    mapk1 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=1)
    mapk5 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=5)

    rows = [
        ("Color+light block", mapk1, mapk5),
    ]

    print(f"\nResults (grid={GRID}, ab_bins={AB_BINS}, l_bins={L_BINS}, {METRIC}):")
    sep = "-" * 38
    print(sep)
    print(f"{'':<18}{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)

    for name, k1, k5 in rows:
        print(f"{name:<18}{k1:>10.4f}{k5:>10.4f}")

    print(sep)

    # List the queries whose correct match is not the top-1 result.
    print("\nFailed queries (ground truth not at rank 1):")
    for qid, retrieved in zip(query_ids, topk5):
        expected = ground_truth[qid]
        if retrieved[0] not in expected:
            rank = next((i + 1 for i, r in enumerate(retrieved) if r in expected), None)
            found = f"rank {rank}" if rank else "not in top-5"
            print(f"  query {qid}: expected {expected}, retrieved top-5 {retrieved} ({found})")
