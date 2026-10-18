import time
import cv2
import numpy as np
import optuna
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from w2.utils import segment, metrics

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
QSD2 = DATA_DIR / "qsd2_w2"
OUT_DIR = Path(__file__).resolve().parent / "results/segmentation"
N_TRIALS = 2000

# Metric to optimize: "f1" | "iou" | "error_rate"  (error_rate is minimized)
METRIC = "iou"
DIRECTION = "minimize" if METRIC == "error_rate" else "maximize"

CHANNELS = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

# Load images and ground-truth masks once
data = []
for path in sorted(QSD2.glob("*.jpg")):
    img = cv2.imread(str(path))
    gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    data.append((img, gt > 0))

assert data, f"No images found in {QSD2}"
print(f"{len(data)} images loaded")


def objective(trial):
    params = {
        "prefilter": trial.suggest_categorical("prefilter", ["none", "median"]),
        "color_space": trial.suggest_categorical("color_space", ["HSV", "LAB"]),
        "channels": CHANNELS[trial.suggest_categorical("channels", list(range(len(CHANNELS))))],
        "threshold": trial.suggest_categorical("threshold", ["otsu", "triangle", "gmm", "wallthr"]),
        "close_frac": trial.suggest_float("close_frac", 0.0, 1.0),
    }
    scores = [metrics(segment(img, **params), gt)[METRIC] for img, gt in data]
    return np.mean(scores)


def log(study, trial):
    print(f"[{trial.number + 1}/{N_TRIALS}] {METRIC}={trial.value:.4f} "
          f"| best={study.best_value:.4f} | {time.time() - t0:.0f}s", end="\r")


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=DIRECTION, sampler=optuna.samplers.TPESampler(seed=0))

t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, callbacks=[log])
print()

# Top 10 trials
df = study.trials_dataframe(attrs=("number", "value", "params")).dropna(subset=["value"])
df["params_channels"] = df["params_channels"].map(lambda i: CHANNELS[i])
df = df.sort_values("value", ascending=(DIRECTION == "minimize")).head(10)

print(df.to_string(index=False))

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / f"top10_{METRIC}.csv", index=False)
print(f"Results saved to {OUT_DIR / f'top10_{METRIC}.csv'}")