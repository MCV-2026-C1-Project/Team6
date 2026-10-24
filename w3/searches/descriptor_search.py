import sys
from pathlib import Path

import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import *
import pickle

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QUERY_DIR = DATA_DIR / "qsd1_w3"
CWD = Path(__file__).resolve().parent

APPLY_SHIFT = False
MAX_SHIFT = 0.03
SEED = 42

N_TRIALS = 50

def random_shift(img, max_shift, rng):
    """Desplaça la imatge un nombre aleatori de píxels en x i y (màx. max_shift * mida)."""
    h, w = img.shape[:2]
    dx = int(rng.integers(-int(max_shift * w), int(max_shift * w) + 1))
    dy = int(rng.integers(-int(max_shift * h), int(max_shift * h) + 1))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
db_ids, db_images = load_images(BBDD_DIR)
query_ids, query_images = load_images(QUERY_DIR)

with open(QUERY_DIR / "gt_corresps.pkl", "rb") as f:
    ground_truth = pickle.load(f)

rng = np.random.default_rng(SEED)
if APPLY_SHIFT:
    query_images = [random_shift(im, MAX_SHIFT, rng) for im in query_images]

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
COLOR_SPACES = {
    "HSV": cv2.COLOR_BGR2HSV,
    "Lab": cv2.COLOR_BGR2Lab,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
}
CHANNELS = ["0", "1", "2", "01", "02", "12", "012"]   # "12" -> [1, 2]

VECTOR_METRICS = ["euclidean", "cosine", "cityblock", "braycurtis", "correlation", "canberra"]
HIST_METRICS = ["chi2", "hellinger", "histogram_intersection", "bhattacharyya", "total_variation"]


# -----------------------------------------------------------------------------
# 1. DCT block
# -----------------------------------------------------------------------------
def objective_dct_block(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3, 4, 6, 8])
    n_coef = trial.suggest_int("n_coef", 5, 64)
    skip_dc = trial.suggest_categorical("skip_dc", [True, False])
    metric = trial.suggest_categorical("metric", VECTOR_METRICS)

    params = {
        "descriptor": dct_block_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": 672,
        "kwargs": dict(n_blocks=n_blocks, n_coef=n_coef, skip_dc=skip_dc),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# 2. DCT histogram
# -----------------------------------------------------------------------------
def objective_dct_histogram(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    size = trial.suggest_categorical("size", [256, 384, 512, 672])
    block_size = trial.suggest_categorical("block_size", [8, 16])
    n_coef = trial.suggest_int("n_coef", 4, 15)
    n_regions = trial.suggest_int("n_regions", 1, 4)
    n_bins = trial.suggest_categorical("n_bins", [8, 16])
    skip_dc = trial.suggest_categorical("skip_dc", [True, False])
    metric = trial.suggest_categorical("metric", HIST_METRICS)

    params = {
        "descriptor": dct_histogram_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": size,
        "kwargs": dict(block_size=block_size, n_coef=n_coef, n_regions=n_regions,
                       n_bins=n_bins, skip_dc=skip_dc),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# 3. DCT pyramid
# -----------------------------------------------------------------------------
def objective_dct_pyramid(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    levels = trial.suggest_categorical("levels", ["1,2", "1,2,3", "1,2,4", "1,2,3,4", "1,2,4,8"])
    n_coef = trial.suggest_int("n_coef", 5, 32)
    skip_dc = trial.suggest_categorical("skip_dc", [True, False])
    metric = trial.suggest_categorical("metric", VECTOR_METRICS)

    params = {
        "descriptor": dct_pyramid_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": 672,
        "kwargs": dict(levels=[int(n) for n in levels.split(",")],
                       n_coef=n_coef, skip_dc=skip_dc),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# 4. DWT block
# -----------------------------------------------------------------------------
def objective_dwt_block(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    size = trial.suggest_categorical("size", [256, 384, 512, 672])
    n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3, 4, 6])
    wavelet = trial.suggest_categorical("wavelet", ["haar", "db2"])
    levels = trial.suggest_int("levels", 1, 3)
    metric = trial.suggest_categorical("metric", VECTOR_METRICS)

    params = {
        "descriptor": dwt_block_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": size,
        "kwargs": dict(n_blocks=n_blocks, wavelet=wavelet, levels=levels),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# 5. DWT histogram
# -----------------------------------------------------------------------------
def objective_dwt_histogram(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    size = trial.suggest_categorical("size", [256, 384, 512, 672])
    n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3, 4, 6])
    wavelet = trial.suggest_categorical("wavelet", ["haar", "db2"])
    levels = trial.suggest_int("levels", 1, 3)
    n_bins = trial.suggest_categorical("n_bins", [8, 16])
    metric = trial.suggest_categorical("metric", HIST_METRICS)

    params = {
        "descriptor": dwt_histogram_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": size,
        "kwargs": dict(n_blocks=n_blocks, wavelet=wavelet, levels=levels, n_bins=n_bins),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# 6. LBP block
# -----------------------------------------------------------------------------
LBP_SCALES = {
    "8_1": ((8, 1.0),),
    "16_2": ((16, 2.0),),
    "8_1+16_2": ((8, 1.0), (16, 2.0)),
    "8_1+16_2+24_3": ((8, 1.0), (16, 2.0), (24, 3.0)),
}

def objective_lbp_block(trial):
    color = trial.suggest_categorical("color", list(COLOR_SPACES))
    channels = trial.suggest_categorical("channels", CHANNELS)
    size = trial.suggest_categorical("size", [256, 384, 512, 672])
    n_blocks = trial.suggest_categorical("n_blocks", [1, 2, 3, 4, 6, 8])
    scales = trial.suggest_categorical("scales", list(LBP_SCALES))
    metric = trial.suggest_categorical("metric", HIST_METRICS)

    params = {
        "descriptor": lbp_block_descriptor,
        "color_code": COLOR_SPACES[color],
        "channels": [int(c) for c in channels],
        "size": size,
        "kwargs": dict(n_blocks=n_blocks, scales=LBP_SCALES[scales], method="uniform"),
    }
    db = np.stack([describe(im, params) for im in db_images])
    q = np.stack([describe(im, params) for im in query_images])

    topk = retrieve(get_distances(q, db, metric), db_ids, k=5)
    trial.set_user_attr("mAP@1", mapk(ground_truth, topk, k=1))
    return mapk(ground_truth, topk, k=5)


# -----------------------------------------------------------------------------
# A bayesian search for each descriptor
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    searches = [
        ("dct_block", objective_dct_block),
        ("dct_histogram", objective_dct_histogram),
        ("dct_pyramid", objective_dct_pyramid),
        ("dwt_block", objective_dwt_block),
        ("dwt_histogram", objective_dwt_histogram),
        ("lbp_block", objective_lbp_block),
    ]

    summary = []
    for name, objective in searches:
        print(f"\n===== {name} =====")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS)

        study.trials_dataframe().to_csv(CWD / "results" / f"bayes_{name}.csv", index=False)

        best = study.best_trial
        print(f"Millor mAP@5: {best.value:.4f} | mAP@1: {best.user_attrs['mAP@1']:.4f}")
        print(f"Paràmetres: {best.params}")
        summary.append({"descriptor": name, "mAP@5": best.value,
                        "mAP@1": best.user_attrs["mAP@1"], **best.params})

    summary = pd.DataFrame(summary).sort_values("mAP@5", ascending=False)
    summary.to_csv(CWD / "results" / "bayes_summary.csv", index=False)
    print("\n", summary.to_string(index=False))