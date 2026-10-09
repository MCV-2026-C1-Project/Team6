from utils import *
import pickle


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w2"

# Descriptor and distance configuration
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
    db_desc = np.array([color_light_block_descriptor(img, GRID, AB_BINS, L_BINS) for img in db_images])
    query_desc = np.array([color_light_block_descriptor(img, GRID, AB_BINS, L_BINS) for img in query_images])

    dists = get_distances(query_desc, db_desc, metric=METRIC)

    # Retrieve the top-5 results and compute mAP@1 and mAP@5.
    topk5 = retrieve(dists, db_ids, k=5)
    mapk1 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=1)
    mapk5 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=5)

    print(f"\ncolor_light_block_descriptor (grid={GRID}, ab_bins={AB_BINS}, l_bins={L_BINS}), {METRIC}")
    sep = "-" * 32
    print(sep)
    print(f"{'mAP@1':>10}{'mAP@5':>10}")
    print(sep)
    print(f"{mapk1:>10.4f}{mapk5:>10.4f}")
    print(sep)
