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
DESC_PARAMS_2D_BEST1 = {"grid": 4, "ab_bins": 48, "l_bins": 16}
DESC_PARAMS_2D_BEST2 = {"grid": 6, "ab_bins": 24, "l_bins": 4}
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
DESC_PARAMS = {"2d": DESC_PARAMS_2D_BEST1, "3d": DESC_PARAMS_3D}[DESCRIPTOR]

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

    gt_masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in sorted(QSD2.glob("*.png"))]
    seg = [metrics(pred, gt) for pred, gt in zip(q2_masks, gt_masks)]

    print("\nTasks 3 & 4: Segmentation QSD2")
    print(sep)
    for key in ["precision", "recall", "f1", "iou", "error_rate"]:
        print(f"{key:<20}{np.mean([r[key] for r in seg]):>10.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # Task 5: retrieval on QSD1 and QSD2 — descriptor × mask comparison.
    # ------------------------------------------------------------------
    with open(QSD2 / "gt_corresps.pkl", "rb") as f:
        gt_q2 = pickle.load(f)

    DESCRIPTORS = {
        "2D BEST QSD1": (color_light_block_descriptors, DESC_PARAMS_2D_BEST1),
        "2D BEST QSD2": (color_light_block_descriptors, DESC_PARAMS_2D_BEST2),
    }
    MASK_SETS = {
        "Without masks": None,
        "Our masks": q2_masks,
        "GT masks": gt_masks,
    }

    task5 = []
    for desc_name, (desc_fn, desc_params) in DESCRIPTORS.items():
        db_desc_t5 = desc_fn(db_images, **desc_params)

        # QSD1 (no masks needed)
        q_desc = desc_fn(q1_images, **desc_params)
        dists = get_distances(q_desc, db_desc_t5, metric=METRIC)
        res = dict(zip(q1_ids, retrieve(dists, db_ids, k=5)))
        task5.append((desc_name, "QSD1", "Without masks",
                      evaluate(gt_q1, res, k=1),
                      evaluate(gt_q1, res, k=5)))

        # QSD2 with each mask set
        for mask_name, masks in MASK_SETS.items():
            if masks is None:
                queries = q2_images
            else:
                queries = [crop_to_mask(img, m) for img, m in zip(q2_images, masks)]

            q_desc = desc_fn(queries, **desc_params)
            dists = get_distances(q_desc, db_desc_t5, metric=METRIC)
            res = dict(zip(q2_ids, retrieve(dists, db_ids, k=5)))

            task5.append((desc_name, "QSD2", mask_name,
                          evaluate(gt_q2, res, k=1),
                          evaluate(gt_q2, res, k=5)))

    # Results table
    sep5 = "-" * 70
    print("\nTask 5: Retrieval QSD1 & QSD2")
    print(sep5)
    print(f"{'Descriptor':<16}{'Dataset':<10}{'Masks':<16}{'mAP@1':>14}{'mAP@5':>14}")
    print(sep5)
    for desc_name, dataset, mask_name, m1, m5 in task5:
        print(f"{desc_name:<16}{dataset:<10}{mask_name:<16}{m1:>14.4f}{m5:>14.4f}")
    print(sep5)

    # ------------------------------------------------------------------
    # Test: QST1 (no masks) and QST2 (masks), top-10.
    # ------------------------------------------------------------------

    db_desc = get_descriptors(db_images, masks=None, **DESC_PARAMS)

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
    for i, (qid, mask) in enumerate(zip(qt2_ids, qt2_masks)):
        stem = Path(str(qid)).stem
        idx = int(stem) if stem.isdigit() else i
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_qt2 / f"{idx:05d}.png"), mask_u8)

    print(f"\nTest results saved to {OUT_DIR}")