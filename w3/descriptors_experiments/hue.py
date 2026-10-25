import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # w3/ for utils.py
from utils import *
import pickle
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w3" / "non_augmented"

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

CHANNEL_SETS = {
    "Y": [0],
    "Cr+Cb": [1, 2],
    "Y+Cr+Cb": [0, 1, 2],
}

MAX_HUE_SHIFT = 40  # degrees
SEED = 42


def random_hue_shift(img, max_deg, rng):
    """Gira el hue de la imatge un angle aleatori entre -max_deg i max_deg graus."""
    shift = int(rng.integers(-max_deg // 2, max_deg // 2 + 1))  # OpenCV: 1 unitat = 2 graus
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QUERY_DIR)

    with open(QUERY_DIR.parent / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    rng = np.random.default_rng(SEED)
    hue_images = [random_hue_shift(im, MAX_HUE_SHIFT, rng) for im in query_images]

    sep = "-" * 52
    print(f"\nHue shift: random, max ±{MAX_HUE_SHIFT} deg")
    print(sep)
    print(f"{'Channels':<10}{'Dim':>6}{'mAP@1':>9}{'mAP@5':>9}{'mAP@1 hue':>11}{'mAP@5 hue':>11}")
    print(sep)

    for name, channels in CHANNEL_SETS.items():
        params = {**PARAMS, "channels": channels}

        db_descs = np.vstack([describe(img, params) for img in tqdm(db_images, desc=f"{name} BBDD", leave=False)])

        results = []
        for images in (query_images, hue_images):
            query_descs = np.vstack([describe(img, params) for img in images])
            dists = get_distances(query_descs, db_descs, metric=params["metric"])
            topk5 = retrieve(dists, db_ids, k=5)
            results += [mapk(ground_truth, topk5, k=1), mapk(ground_truth, topk5, k=5)]

        print(f"{name:<10}{db_descs.shape[1]:>6}" + "".join(f"{r:>{w}.4f}" for r, w in zip(results, (9, 9, 11, 11))))

    print(sep)