from utils import *
import pickle
import pandas as pd


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
OUT_DIR = Path(__file__).resolve().parent / "results"

week = "w2"
QSD1 = DATA_DIR / f"qsd1_{week}"
QSD2 = DATA_DIR / f"qsd2_{week}"
QST1 = DATA_DIR / f"qst1_{week}"
QST2 = DATA_DIR / f"qst2_{week}"


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
    gt_paths = sorted(QSD2.glob("*.png"))
    gt_masks = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for path in gt_paths
    ]

    results = [
        metrics(pred, gt)
        for pred, gt in zip(pred_masks, gt_masks)
    ]

    print("\nSegmentation results")
    print("-" * 50)
    print(f"Precision:  {np.mean([r['precision'] for r in results]):.4f}")
    print(f"Recall:     {np.mean([r['recall'] for r in results]):.4f}")
    print(f"F1:         {np.mean([r['f1'] for r in results]):.4f}")
    print(f"IoU:        {np.mean([r['iou'] for r in results]):.4f}")
    print(f"Error rate: {np.mean([r['error_rate'] for r in results]):.4f}")
    print("-" * 50)


def test():
    print("Loading images...")
    db_ids, db_images = load_images(BBDD_DIR)

    descriptor_params = BEST_PARAMS["2d_descriptor"]
    segmentation_params = BEST_PARAMS["segmentation"]

    # Segment database images
    print("Segmenting database images...")
    db_masks = segments(db_images, **segmentation_params)

    # Compute masked database descriptors
    print("Extracting database descriptors...")
    db_desc = color_light_block_descriptors(db_images, masks=db_masks, **descriptor_params)

    for query_dir in [QST1, QST2]:
        print(f"\nProcessing {query_dir.name}...")

        query_ids, query_images = load_images(query_dir)

        # Segment query images
        query_masks = segments(query_images, **segmentation_params)

        # Compute masked query descriptors
        query_desc = color_light_block_descriptors(query_images, masks=query_masks, **descriptor_params)

        # Retrieval
        dists = get_distances(query_desc, db_desc, metric=METRIC)
        results = retrieve(dists, db_ids, k=10)

        print(f"Results for {query_dir.name}:")
        print(results)

        # Save result.pkl
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(OUT_DIR / "result.pkl", "wb") as f:
            pickle.dump(results, f)

        # QST2: save masks
        if query_dir == QST2:
            for query_id, mask in zip(query_ids, query_masks):
                cv2.imwrite(
                    str(OUT_DIR / f"masks/{query_id}.png"),
                    (mask * 255).astype(np.uint8)
                )

        print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    # Task 1 & 2
    # retrieval_task(QSD1, "2d_descriptor", use_masks=False)

    # Task 3 & 4
    segmentation()

    # Task 5
    # retrieval_task(QSD2, "2d_descriptor", use_masks=True)