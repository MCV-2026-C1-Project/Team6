from utils import *
import pickle


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w2"
QSD2 = DATA_DIR / "qsd2_w2"
QST1 = DATA_DIR / "qst1_w2"
QST2 = DATA_DIR / "qst2_w2"
OUT_DIR = Path(__file__).resolve().parent / "results"

METRIC = "histogram_intersection"
DESCRIPTOR = "2d"

# Best parameters
DESC_PARAMS_2D = {"grid": 4, "ab_bins": 48, "l_bins": 16}
DESC_PARAMS_3D = {"bins": (2, 8, 8), "grids": (4,)}
SEG_PARAMS = {
    "prefilter": "none",
    "color_space": "HSV",
    "channels": [1, 2],
    "threshold": "otsu",
    "close_frac": 0.012306,
}

# Selected descriptor
get_descriptors = {"2d": color_light_block_descriptors, "3d": spatial_descriptors_3d}[DESCRIPTOR]
DESC_PARAMS = {"2d": DESC_PARAMS_2D, "3d": DESC_PARAMS_3D}[DESCRIPTOR]

if __name__ == "__main__":
    sep = "-" * 40
    print(f"Descriptor: {DESCRIPTOR}")

    # Load database.
    db_ids, db_images = load_images(BBDD_DIR)

    # ------------------------------------------------------------------
    # Tasks 1 & 2: retrieval on QSD1 without masks.
    # ------------------------------------------------------------------
    q1_ids, q1_images = load_images(QSD1)
    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        gt_q1 = pickle.load(f)

    db_desc = get_descriptors(db_images, masks=None, **DESC_PARAMS)
    q1_desc = get_descriptors(q1_images, masks=None, **DESC_PARAMS)

    dists_q1 = get_distances(q1_desc, db_desc, metric=METRIC)
    results_q1 = dict(zip(q1_ids, retrieve(dists_q1, db_ids, k=5)))

    print("\nTasks 1 & 2: Retrieval QSD1 (no masks)")
    print(sep)
    print(f"{'mAP@1':<20}{evaluate(gt_q1, results_q1, k=1):>10.4f}")
    print(f"{'mAP@5':<20}{evaluate(gt_q1, results_q1, k=5):>10.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # Tasks 3 & 4: background removal on QSD2.
    # ------------------------------------------------------------------
    q2_ids, q2_images = load_images(QSD2)
    q2_masks = segments(q2_images, **SEG_PARAMS)

    gt_masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(QSD2.glob("*.png"))]
    seg = [metrics(pred, gt) for pred, gt in zip(q2_masks, gt_masks)]

    print("\nTasks 3 & 4: Segmentation QSD2")
    print(sep)
    for key in ["precision", "recall", "f1", "iou", "error_rate"]:
        print(f"{key:<20}{np.mean([r[key] for r in seg]):>10.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # Task 5: retrieval on QSD2 with masks.
    # ------------------------------------------------------------------
    with open(QSD2 / "gt_corresps.pkl", "rb") as f:
        gt_q2 = pickle.load(f)

    crops = [crop_to_mask(img, m) for img, m in zip(q2_images, q2_masks)]
    q2_desc = get_descriptors(crops, masks=None, **DESC_PARAMS)

    dists_q2 = get_distances(q2_desc, db_desc, metric=METRIC)
    results_q2 = dict(zip(q2_ids, retrieve(dists_q2, db_ids, k=5)))

    print("\nTask 5: Retrieval QSD2 (masks)")
    print(sep)
    print(f"{'mAP@1':<20}{evaluate(gt_q2, results_q2, k=1):>10.4f}")
    print(f"{'mAP@5':<20}{evaluate(gt_q2, results_q2, k=5):>10.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # Test: QST1 (no masks) and QST2 (masks), top-10.
    # ------------------------------------------------------------------
    # QST1
    qt1_ids, qt1_images = load_images(QST1)
    qt1_desc = get_descriptors(qt1_images, **DESC_PARAMS)
    results_qt1 = retrieve(get_distances(qt1_desc, db_desc, metric=METRIC), db_ids, k=10)

    out_qt1 = OUT_DIR / "QST1" / "method1"
    out_qt1.mkdir(parents=True, exist_ok=True)
    with open(out_qt1 / "result.pkl", "wb") as f:
        pickle.dump(results_qt1, f)

    # QST2
    qt2_ids, qt2_images = load_images(QST2)
    qt2_masks = segments(qt2_images, **SEG_PARAMS)

    crops = [crop_to_mask(img, m) for img, m in zip(qt2_images, qt2_masks)]
    qt2_desc = get_descriptors(crops, masks=None, **DESC_PARAMS)
    results_qt2 = retrieve(get_distances(qt2_desc, db_desc, metric=METRIC), db_ids, k=10)

    out_qt2 = OUT_DIR / "QST2" / "method1"
    out_qt2.mkdir(parents=True, exist_ok=True)
    with open(out_qt2 / "result.pkl", "wb") as f:
        pickle.dump(results_qt2, f)
    for qid, mask in zip(qt2_ids, qt2_masks):
        cv2.imwrite(str(out_qt2 / f"{qid}.png"), (mask * 255).astype(np.uint8))

    print(f"\nTest results saved to {OUT_DIR}")