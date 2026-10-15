"""
Cerca bayesiana (Optuna, TPE) del pipeline de segmentació de quadres sobre paret
(sense detecció de vores).

Pipeline (per a cada imatge):
    1. Prefilter         : none | median (kernel MEDIAN_KSIZE)
    2. Color space       : HSV | LAB, amb un subconjunt de canals
    3. Distància de Mahalanobis respecte el fons (franja de BORDER_SIZE px)
    4. Threshold         : otsu | triangle | gmm | wallthr
    5. Close             : kernel quadrat n = close_frac * min(h, w), close_frac ∈ [0, 1] (float)
    6. Component connexa més gran + fill holes
    7. force_rect        : rectangle a partir de projeccions de files/columnes (PROJ_FRAC)

Requisits: pip install optuna opencv-python numpy pandas scipy scikit-learn

Ús:
    python bayes_search_segmentation.py
    python bayes_search_segmentation.py --data ../data/qsd2_w2 --trials 300 --metric iou --jobs 8

Sortida (a OUT_DIR):
    trials.csv            -> totes les proves, ordenades per METRIC
    per_image_best.csv    -> mètriques per imatge de la millor configuració
    best_masks/*.png      -> màscares de la millor configuració (si SAVE_BEST_MASKS)
"""

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import optuna
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "qsd2_w2"  # *.jpg + *.png (GT)
OUT_DIR = Path(__file__).resolve().parent / "bayes_search_out"

# Mètrica a optimitzar: "f1" | "iou" | "error_rate"  (error_rate es minimitza)
METRIC = "iou"

N_TRIALS = 2000             # nombre de configuracions que prova la cerca bayesiana
SEED = 0                    # llavor del sampler TPE (reproduïbilitat)
N_JOBS = os.cpu_count()     # processos en paral·lel (per imatges, dins de cada prova)
# Opcional: base de dades per poder reprendre la cerca, p. ex. "sqlite:///bayes.db"
STORAGE = None
STUDY_NAME = "painting_segmentation"

BORDER_SIZE = 15            # px de vora per modelar la paret
MEDIAN_KSIZE = 5            # mida del kernel del median (si prefilter = "median")
SAVE_BEST_MASKS = True

# ---- Espai de cerca ----
PREFILTERS = ["none", "median"]
COLOR_SPACES = ["HSV", "LAB"]
# Subconjunts de canals (índexs 0,1,2). Per defecte: tots els subconjunts no buits.
#   HSV: 0=H, 1=S, 2=V      LAB: 0=L, 1=a, 2=b
CHANNEL_SETS = ["".join(map(str, c)) for r in (1, 2, 3)
                for c in itertools.combinations(range(3), r)]   # "0", "1", ..., "012"
THRESHOLDS = ["otsu", "triangle", "gmm", "wallthr"]
CLOSE_FRAC_RANGE = (0.0, 1.0)  # rang float de close_frac

# Paràmetres fixos dels mètodes de threshold (els dels notebooks)
WALLTHR_K = 6               # t = mediana + K * 1.4826 * MAD
GMM_SAMPLE_SIZE = 14        # la imatge es redueix a 14x14 per ajustar la GMM
GMM_SMOOTH_SIGMA = 2

# Memòria cau (per procés) de les màscares binàries del pas 4, per no recalcular
# Mahalanobis + threshold quan TPE repeteix la mateixa part categòrica.
# Cada entrada ocupa h*w bytes (una imatge). Baixa-ho si et quedes sense RAM.
FG_CACHE_SIZE = 256

# Pas 7 (force_rect): una fila/columna és quadre si té més de PROJ_FRAC del màxim
PROJ_FRAC = 0.5
# =============================================================================

CHANNEL_NAMES = {"HSV": "HSV", "LAB": "Lab"}
CVT = {"HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB}


# -----------------------------------------------------------------------------
# Passos del pipeline
# -----------------------------------------------------------------------------
def prefilter(bgr, name):
    if name == "none":
        return bgr
    if name == "median":
        return cv2.medianBlur(bgr, MEDIAN_KSIZE)
    raise ValueError(f"Prefilter desconegut: {name}")


def border_mask(h, w, b=BORDER_SIZE):
    b = max(1, min(b, h // 2, w // 2))
    m = np.zeros((h, w), bool)
    m[:b, :] = m[-b:, :] = True
    m[:, :b] = m[:, -b:] = True
    return m


def mahalanobis(color, border, channels):
    """Distància de Mahalanobis de cada píxel a la distribució dels píxels de la vora."""
    x = color[..., channels].astype(np.float64)
    bg = x[border]
    mean = bg.mean(axis=0)
    cov = np.atleast_2d(np.cov(bg, rowvar=False)) + np.eye(len(channels)) * 1e-3
    icov = np.linalg.pinv(cov)
    delta = x - mean
    d2 = np.einsum("...i,ij,...j->...", delta, icov, delta)
    return np.sqrt(np.maximum(d2, 0)).astype(np.float32)


def threshold_value(distance, border, method):
    """Retorna el llindar t (en unitats de distància). Foreground = distance > t."""
    if method == "wallthr":
        wall_d = distance[border].astype(np.float64)
        med = np.median(wall_d)
        return med + WALLTHR_K * 1.4826 * np.median(np.abs(wall_d - med))

    # otsu / triangle / gmm treballen sobre la distància en uint8 (retallada a [0, 255])
    src = np.clip(distance, 0, 255).astype(np.uint8)

    if method == "otsu":
        t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return t
    if method == "triangle":
        t, _ = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return t
    if method == "gmm":
        sample = cv2.resize(src, (GMM_SAMPLE_SIZE, GMM_SAMPLE_SIZE),
                            interpolation=cv2.INTER_CUBIC).reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(sample)
        m1, m2 = np.sort(gmm.means_.flatten())
        hist, bins = np.histogram(src, bins=256, range=(0, 256))
        hist_smooth = gaussian_filter1d(hist.astype(np.float64), sigma=GMM_SMOOTH_SIGMA)
        centers = (bins[:-1] + bins[1:]) / 2
        between = (centers > m1) & (centers < m2)
        if not between.any():           # les dues gaussianes massa juntes
            return (m1 + m2) / 2
        return centers[between][np.argmin(hist_smooth[between])]
    raise ValueError(f"Threshold desconegut: {method}")


def close(mask, frac):
    h, w = mask.shape
    n = int(frac * min(h, w))
    if n < 2:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def largest_component_filled(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ndi.binary_fill_holes(labels == largest).astype(np.uint8)


def force_rect(mask):
    """Rectangle alineat amb els eixos a partir de les projeccions de files i columnes.
    Una columna (fila) és quadre si la seva fracció de foreground supera
    PROJ_FRAC * el màxim de totes les columnes (files)."""
    H, W = mask.shape
    mask_rect = np.zeros((H, W), np.uint8)
    if not mask.any():
        return mask_rect
    col_prof, row_prof = mask.mean(axis=0), mask.mean(axis=1)
    cols = col_prof > PROJ_FRAC * col_prof.max()
    rows = row_prof > PROJ_FRAC * row_prof.max()
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    mask_rect[y0:y1 + 1, x0:x1 + 1] = 1
    return mask_rect


def foreground(bgr, pf, cs, ch, th):
    """Passos 1-4: retorna (màscara binària, llindar)."""
    h, w = bgr.shape[:2]
    border = border_mask(h, w)
    color = cv2.cvtColor(prefilter(bgr, pf), CVT[cs])
    dist = mahalanobis(color, border, [int(c) for c in ch])
    t = threshold_value(dist, border, th)
    return (dist > t).astype(np.uint8), float(t)


def predict(bgr, pf, cs, ch, th, cf):
    fg, _ = foreground(bgr, pf, cs, ch, th)
    mask = largest_component_filled(close(fg, cf))
    return force_rect(mask)


# -----------------------------------------------------------------------------
# Mètriques
# -----------------------------------------------------------------------------
def metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "error_rate": (fp + fn) / gt.size,
    }


# -----------------------------------------------------------------------------
# Treballadors (cada procés carrega les imatges una sola vegada)
# -----------------------------------------------------------------------------
_IMAGES = []   # [(stem, bgr, gt)]


def _init_worker(pairs):
    global _IMAGES
    _IMAGES = []
    for img_path, gt_path in pairs:
        bgr = cv2.imread(str(img_path))
        gt = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if gt.shape != bgr.shape[:2]:
            gt = cv2.resize(gt, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        _IMAGES.append((img_path.stem, bgr, gt))


@lru_cache(maxsize=FG_CACHE_SIZE)
def _cached_foreground(idx, pf, cs, ch, th):
    fg, t = foreground(_IMAGES[idx][1], pf, cs, ch, th)
    fg.setflags(write=False)
    return fg, t


def _eval_one(task):
    idx, (pf, cs, ch, th, cf) = task
    stem, _, gt = _IMAGES[idx]
    fg, t = _cached_foreground(idx, pf, cs, ch, th)
    pred = force_rect(largest_component_filled(close(fg, cf)))
    return {"image": stem, "thr_value": t, **metrics(pred, gt)}


# -----------------------------------------------------------------------------
# Cerca bayesiana
# -----------------------------------------------------------------------------
MET_COLS = ["precision", "recall", "f1", "iou", "error_rate"]


def evaluate_config(pool, n_images, cfg, n_jobs):
    """Avalua una configuració sobre totes les imatges; retorna DataFrame per imatge."""
    tasks = [(i, cfg) for i in range(n_images)]
    chunk = max(1, n_images // (4 * max(1, n_jobs)))
    return pd.DataFrame(list(pool.map(_eval_one, tasks, chunksize=chunk)))


def suggest_config(trial):
    return (
        trial.suggest_categorical("prefilter", PREFILTERS),
        trial.suggest_categorical("color_space", COLOR_SPACES),
        trial.suggest_categorical("channels", CHANNEL_SETS),
        trial.suggest_categorical("threshold", THRESHOLDS),
        trial.suggest_float("close_frac", *CLOSE_FRAC_RANGE),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--metric", choices=["f1", "iou", "error_rate"], default=METRIC)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    args = ap.parse_args()

    pairs = [(p, p.with_suffix(".png")) for p in sorted(args.data.glob("*.jpg"))
             if p.with_suffix(".png").exists()]
    assert pairs, f"No s'han trobat parelles .jpg/.png a {args.data}"
    print(f"{len(pairs)} imatges | {args.trials} proves | mètrica: {args.metric}")

    direction = "minimize" if args.metric == "error_rate" else "maximize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        storage=STORAGE, study_name=STUDY_NAME, load_if_exists=STORAGE is not None,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=(pairs,)) as pool:

        def objective(trial):
            per_img = evaluate_config(pool, len(pairs), suggest_config(trial), args.jobs)
            means = per_img[MET_COLS].mean()
            for m in MET_COLS:
                trial.set_user_attr(f"{m}_mean", float(means[m]))
            trial.set_user_attr(f"{args.metric}_min" if direction == "maximize"
                                else f"{args.metric}_max",
                                float(per_img[args.metric].min() if direction == "maximize"
                                      else per_img[args.metric].max()))
            return float(means[args.metric])

        def log(study, trial):
            print(f"  [{trial.number + 1}/{args.trials}] {args.metric}={trial.value:.4f} "
                  f"| millor={study.best_value:.4f} | {time.time() - t0:.0f}s", end="\r")

        study.optimize(objective, n_trials=args.trials, callbacks=[log])
        print()

        # ---- Resultats ----
        args.out.mkdir(parents=True, exist_ok=True)
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        df.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in df.columns]
        df = df.sort_values("value", ascending=direction == "minimize")
        df.to_csv(args.out / "trials.csv", index=False, float_format="%.5f")

        bp = study.best_trial.params
        best_cfg = (bp["prefilter"], bp["color_space"], bp["channels"],
                    bp["threshold"], bp["close_frac"])
        per_img = evaluate_config(pool, len(pairs), best_cfg, args.jobs)
        per_img = per_img.sort_values(args.metric, ascending=direction == "maximize")
        per_img.to_csv(args.out / "per_image_best.csv", index=False, float_format="%.5f")

    pd.set_option("display.width", 200)
    df["channels"] = [ "".join(CHANNEL_NAMES[cs][int(c)] for c in ch)
                       for cs, ch in zip(df["color_space"], df["channels"]) ]
    show = ["number", "prefilter", "color_space", "channels", "threshold", "close_frac",
            "f1_mean", "iou_mean", "error_rate_mean"]
    print(f"\nTop 15 proves (per {args.metric} mitjà):")
    print(df[show].head(15).to_string(index=False))

    pf, cs, ch, th, cf = best_cfg
    ch_name = "".join(CHANNEL_NAMES[cs][int(c)] for c in ch)
    print(f"\nMillor: prefilter={pf} color_space={cs} channels={ch_name} "
          f"threshold={th} close_frac={cf:.4f} -> {args.metric}={study.best_value:.4f}")
    print("Pitjors imatges amb la millor configuració:")
    print(per_img[["image", "thr_value"] + MET_COLS].head(5).to_string(index=False))

    if SAVE_BEST_MASKS:
        mdir = args.out / "best_masks"
        mdir.mkdir(exist_ok=True)
        for img_path, _ in pairs:
            m = predict(cv2.imread(str(img_path)), *best_cfg)
            cv2.imwrite(str(mdir / f"{img_path.stem}.png"), m * 255)
        print(f"\nMàscares guardades a {mdir}")
    print(f"Resultats a {args.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()