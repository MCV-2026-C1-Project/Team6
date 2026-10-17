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

def retrieval_task(query_dir, descriptor="2d_descriptor", use_masks=False):
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(query_dir)

    desc_params = BEST_PARAMS[descriptor]
    descriptor_function = {
        "2d_descriptor": color_light_block_descriptors,
        "3d_descriptor": spatial_descriptors_3d,
    }[descriptor]

    if use_masks:
        seg_params = BEST_PARAMS["segmentation"]
        db_masks = segments(db_images, **seg_params)
        query_masks = segments(query_images, **seg_params)
    else:
        db_masks = query_masks = None

    db_desc = descriptor_function(db_images, masks=db_masks, **desc_params)
    query_desc = descriptor_function(query_images, masks=query_masks, **desc_params)

    dists = get_distances(query_desc, db_desc, metric=METRIC)
    topk5 = retrieve(dists, db_ids, k=5)
    results = dict(zip(query_ids, topk5))

    with open(query_dir / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    mapk1 = evaluate(ground_truth, results, k=1)
    mapk5 = evaluate(ground_truth, results, k=5)

    print(f"\nResults ({descriptor})")
    print("-" * 50)
    print(f"{'':<25}{'mAP@1':>10}{'mAP@5':>10}")
    print("-" * 50)
    print(f"{descriptor:<25}{mapk1:>10.4f}{mapk5:>10.4f}")
    print("-" * 50)

    print("\nFailed queries:")
    for qid, retrieved in results.items():
        expected = ground_truth[qid]
        if retrieved[0] not in expected:
            rank = next((i + 1 for i, r in enumerate(retrieved) if r in expected), None)
            print(
                f"  {qid}: expected {expected}, retrieved {retrieved} "
                f"({'rank ' + str(rank) if rank else 'not in top-5'})"
            )

    return results

def segmentation():
    query_ids, query_images = load_images(QSD2)

    # Predicted masks
    pred_masks = segments(query_images, **BEST_PARAMS["segmentation"])

    # Ground-truth masks
    gt_masks = [
        cv2.imread(str(QSD2 / f"{qid}.png"), cv2.IMREAD_GRAYSCALE)
        for qid in query_ids
    ]

    precisions, recalls, f1s = [], [], []

    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred > 0
        gt = gt > 0

        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("\nSegmentation results")
    print("-" * 50)
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1:        {np.mean(f1s):.4f}")
    print("-" * 50)
    return pred_masks


if __name__ == "__main__":
    # Task 1 & 2
    retrieval_task(QSD1, "2d_descriptor", use_masks=False)

    # Task 3 & 4
    pred_masks = segmentation()

    # Task 5
    retrieval_task(QSD2, "2d_descriptor", use_masks=True)