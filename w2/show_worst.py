"""
Donats uns hiperparàmetres, executa el pipeline sobre totes les imatges i mostra
les N_WORST pitjors segons METRIC i guarda un collage (SAVE_PATH) amb els passos intermedis:

    original | distància Mahalanobis | threshold | close | component + fill | rect | comparació amb GT
    (comparació: verd = TP, vermell = FP, blau = FN)

Reutilitza les funcions de bayes_search_segmentation.py (ha d'estar a la mateixa carpeta).

Ús:
    python show_worst.py                                   # hiperparàmetres de sota
    python show_worst.py --cs LAB --channels Lab --threshold otsu --close 0.12 --prefilter median
    python show_worst.py --from-csv bayes_search_out/trials.csv   # agafa la millor prova
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")       # no obre finestres, només guarda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayes_search_segmentation as seg

# =============================================================================
# HIPERPARÀMETRES (es poden sobreescriure per línia de comandes)
# =============================================================================
PREFILTER = "none"          # "none" | "median"  (mida: seg.MEDIAN_KSIZE)
COLOR_SPACE = "HSV"         # "HSV" | "LAB"
CHANNELS = "SV"             # noms dels canals: HSV -> H,S,V   LAB -> L,a,b
THRESHOLD = "otsu"          # "otsu" | "triangle" | "gmm" | "wallthr"
CLOSE_FRAC = 0.012306          # n = close_frac * min(h, w)

METRIC = "iou"               # "f1" | "iou" | "error_rate"
N_WORST = 10
DATA_DIR = seg.DATA_DIR
SAVE_PATH = Path(__file__).resolve().parent / "worst_examples.png"
# =============================================================================


def parse_channels(cs, channels):
    """Accepta noms ('SV', 'Lab') o índexs ('12', '012')."""
    if channels.isdigit():
        return [int(c) for c in channels]
    names = seg.CHANNEL_NAMES[cs]
    try:
        return [names.index(c) for c in channels]
    except ValueError:
        raise SystemExit(f"Canals '{channels}' no vàlids per a {cs} (opcions: {names})")


def run(bgr, pf, cs, ch, th, cf):
    """Pipeline complet retornant també els passos intermedis."""
    h, w = bgr.shape[:2]
    border = seg.border_mask(h, w)
    color = cv2.cvtColor(seg.prefilter(bgr, pf), seg.CVT[cs])
    dist = seg.mahalanobis(color, border, ch)
    t = seg.threshold_value(dist, border, th)
    fg = (dist > t).astype(np.uint8)
    closed = seg.close(fg, cf)
    filled = seg.largest_component_filled(closed)
    final = seg.force_rect(filled)          # funció de bayes_search_segmentation.py
    return {"dist": dist, "t": float(t), "fg": fg, "closed": closed, "filled": filled,
            "final": final}


def comparison(bgr, pred, gt):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    over = np.zeros_like(rgb)
    pred, gt = pred.astype(bool), gt.astype(bool)
    over[pred & gt] = (0, 200, 0)
    over[pred & ~gt] = (220, 0, 0)
    over[~pred & gt] = (0, 0, 220)
    return cv2.addWeighted(rgb, 0.5, over, 0.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefilter", default=PREFILTER, choices=["none", "median"])
    ap.add_argument("--cs", default=COLOR_SPACE, choices=["HSV", "LAB"])
    ap.add_argument("--channels", default=CHANNELS)
    ap.add_argument("--threshold", default=THRESHOLD, choices=seg.THRESHOLDS)
    ap.add_argument("--close", type=float, default=CLOSE_FRAC)
    ap.add_argument("--from-csv", type=Path, help="trials.csv: fa servir la primera fila (la millor)")
    ap.add_argument("--metric", default=METRIC, choices=["f1", "iou", "error_rate"])
    ap.add_argument("-n", type=int, default=N_WORST)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--save", type=Path, default=SAVE_PATH)
    args = ap.parse_args()

    if args.from_csv:
        row = pd.read_csv(args.from_csv, dtype={"channels": str}).iloc[0]
        args.prefilter, args.cs = row["prefilter"], row["color_space"]
        args.channels, args.threshold, args.close = row["channels"], row["threshold"], row["close_frac"]

    ch = parse_channels(args.cs, args.channels)
    ch_name = "".join(seg.CHANNEL_NAMES[args.cs][i] for i in ch)
    cfg_str = (f"prefilter={args.prefilter} | {args.cs}[{ch_name}] | thr={args.threshold} "
               f"| close={args.close:.4f}")

    pairs = [(p, p.with_suffix(".png")) for p in sorted(args.data.glob("*.jpg"))
             if p.with_suffix(".png").exists()]
    assert pairs, f"No s'han trobat parelles .jpg/.png a {args.data}"

    # ---- Avaluació de totes les imatges ----
    results = []
    for img_path, gt_path in pairs:
        bgr = cv2.imread(str(img_path))
        gt = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if gt.shape != bgr.shape[:2]:
            gt = cv2.resize(gt, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        out = run(bgr, args.prefilter, args.cs, ch, args.threshold, args.close)
        results.append({"image": img_path.stem, "path": img_path, "gt": gt, **out,
                        **seg.metrics(out["final"], gt)})

    worst_first = args.metric != "error_rate"         # f1/iou: valors baixos = pitjors
    order = sorted(range(len(results)), key=lambda i: results[i][args.metric],
                   reverse=not worst_first)[:args.n]

    # ---- Figura ----
    cols = ["Original", "Distància", "Threshold", "Close", "Component + fill", "Rect", "vs GT"]
    fig, axes = plt.subplots(len(order), len(cols), figsize=(3 * len(cols), 3 * len(order)),
                             squeeze=False)
    for row, i in zip(axes, order):
        r = results[i]
        bgr = cv2.imread(str(r["path"]))
        imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), r["dist"], r["fg"], r["closed"],
                r["filled"], r["final"], comparison(bgr, r["final"], r["gt"])]
        for j, (ax, im) in enumerate(zip(row, imgs)):
            if j == 1:
                ax.imshow(im, cmap="magma", vmin=0, vmax=np.percentile(im, 99))
            elif im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(f"{r['image']}\n{args.metric}={r[args.metric]:.3f}", fontsize=10)
        row[2].set_title(f"t = {r['t']:.1f}", fontsize=9)
    for ax, c in zip(axes[0], cols):
        ax.set_title(c if c != "Threshold" else f"{c}\n{ax.get_title()}", fontsize=10)
    fig.suptitle(cfg_str, fontsize=12)
    fig.tight_layout()

    fig.savefig(args.save, dpi=100)
    plt.close(fig)
    print(f"Collage guardat a {args.save}")


if __name__ == "__main__":
    main()