from utils import *
import pickle
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w3"                 # amb noise
CLEAN_DIR = QUERY_DIR / "non_augmented"          # sense noise

MAX_SHIFT = 0.05      # fracció de la mida de la imatge
MAX_HUE_SHIFT = 40    # graus
SEED = 42
DEFAULT_SIZE = 672    # per als descriptors on bayes_summary no fixa "size"

# Millors paràmetres de bayes_summary.csv.
# Comprova que els noms de les funcions i dels kwargs coincideixen amb utils.py.
CONFIGS = {
    "dct_block": {
        "descriptor": dct_block_descriptor,
        "color_code": cv2.COLOR_BGR2YCrCb,
        "channels": [0, 1, 2],
        "size": DEFAULT_SIZE,
        "metric": "cityblock",
        "kwargs": {"n_blocks": 6, "n_coef": 20, "skip_dc": True},
    },
    "dct_pyramid": {
        "descriptor": dct_pyramid_descriptor,
        "color_code": cv2.COLOR_BGR2YCrCb,
        "channels": [0, 1, 2],
        "size": DEFAULT_SIZE,
        "metric": "canberra",
        "kwargs": {"levels": [1, 2, 3], "n_coef": 32, "skip_dc": False},
    },
    "dct_histogram": {
        "descriptor": dct_histogram_descriptor,
        "color_code": cv2.COLOR_BGR2Lab,
        "channels": [0, 2],
        "size": 256,
        "metric": "hellinger",
        "kwargs": {"n_coef": 7, "skip_dc": False, "block_size": 16, "n_regions": 4, "n_bins": 8},
    },
    "dwt_block": {
        "descriptor": dwt_block_descriptor,
        "color_code": cv2.COLOR_BGR2Lab,
        "channels": [0, 2],
        "size": 384,
        "metric": "cosine",
        "kwargs": {"n_blocks": 6, "levels": 3, "wavelet": "db2"},
    },
    "dwt_histogram": {
        "descriptor": dwt_histogram_descriptor,
        "color_code": cv2.COLOR_BGR2YCrCb,
        "channels": [0, 2],
        "size": 256,
        "metric": "hellinger",
        "kwargs": {"n_blocks": 6, "n_bins": 8, "levels": 3, "wavelet": "db2"},
    },
    "lbp_block": {
        "descriptor": lbp_block_descriptor,
        "color_code": cv2.COLOR_BGR2Lab,
        "channels": [0, 1, 2],
        "size": 256,
        "metric": "chi2",
        "kwargs": {"n_blocks": 6, "scales": [(16, 2)]},  # "16_2" -> (P=16, R=2)
    },
}


def random_shift(img, max_shift, rng):
    """Desplaça la imatge un nombre aleatori de píxels en x i y (màx. max_shift * mida)."""
    h, w = img.shape[:2]
    dx = int(rng.integers(-int(max_shift * w), int(max_shift * w) + 1))
    dy = int(rng.integers(-int(max_shift * h), int(max_shift * h) + 1))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_hue_shift(img, max_deg, rng):
    """Gira el hue de la imatge un angle aleatori entre -max_deg i max_deg graus."""
    shift = int(rng.integers(-max_deg // 2, max_deg // 2 + 1))  # OpenCV: 1 unitat = 2 graus
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    db_ids, db_images = load_images(BBDD_DIR)
    _, clean_images = load_images(CLEAN_DIR)
    _, noisy_images = load_images(QUERY_DIR)

    with open(QUERY_DIR / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    # Hue i shift s'apliquen sobre les imatges SENSE noise, cadascun amb el seu rng
    # perquè els resultats no depenguin de l'ordre.
    rng_hue = np.random.default_rng(SEED)
    hue_images = [random_hue_shift(im, MAX_HUE_SHIFT, rng_hue) for im in clean_images]
    rng_shift = np.random.default_rng(SEED)
    shift_images = [random_shift(im, MAX_SHIFT, rng_shift) for im in clean_images]

    cases = {
        "Original": clean_images,
        "Noise": noisy_images,
        f"Hue (±{MAX_HUE_SHIFT}°)": hue_images,
        f"Shift (±{int(MAX_SHIFT * 100)}%)": shift_images,
    }

    rows = []
    for name, params in CONFIGS.items():
        db_descs = np.vstack([describe(img, params) for img in tqdm(db_images, desc=f"{name} BBDD", leave=False)])
        row = {"Descriptor": name}
        for case, images in cases.items():
            query_descs = np.vstack([describe(img, params) for img in images])
            dists = get_distances(query_descs, db_descs, metric=params["metric"])
            topk = retrieve(dists, db_ids, k=1)
            row[case] = mapk(ground_truth, topk, k=1)
        rows.append(row)
        print(f"{name}: " + "  ".join(f"{c}={row[c]:.4f}" for c in cases))

    df = pd.DataFrame(rows).set_index("Descriptor")
    print("\nmAP@1")
    print(df.to_string(float_format="{:.4f}".format))

    df.to_csv("robustness_map1.csv", float_format="%.4f")
    with open("robustness_map1.md", "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))