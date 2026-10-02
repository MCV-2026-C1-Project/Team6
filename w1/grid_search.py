from utils import *
import pickle
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
BBDD_DIR = DATA_DIR / 'BBDD'
QSD1 = DATA_DIR / 'qsd1_w1'

if __name__ == '__main__':
    db_ids, db_images = load_images_lab(BBDD_DIR)
    query_ids, query_images = load_images_lab(QSD1)

    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    db_m1 = simple_descriptors(db_images)
    query_m1 = simple_descriptors(query_images)
    db_m2 = complex_descriptors(db_images)
    query_m2 = complex_descriptors(query_images)

    results = []
    for metric in DISTANCES:
        try:
            # Method 1
            dists_m1 = get_distances(query_m1, db_m1, metric=metric)
            topk5 = retrieve(dists_m1, db_ids, k=5)
            m1_mapk1 = mapk(ground_truth, topk5, k=1)
            m1_mapk5 = mapk(ground_truth, topk5, k=5)

            # Method 2
            dists_m2 = get_distances(query_m2, db_m2, metric=metric)
            topk5 = retrieve(dists_m2, db_ids, k=5)
            m2_mapk1 = mapk(ground_truth, topk5, k=1)
            m2_mapk5 = mapk(ground_truth, topk5, k=5)

        except Exception as e:
            print(f"Skipping {metric}: {e}")

        results.append({
            "metric": metric,
            "m1_mAP@1": m1_mapk1,
            "m1_mAP@5": m1_mapk5,
            "m2_mAP@1": m2_mapk1,
            "m2_mAP@5": m2_mapk5,
            "mean_mAP@1": (m1_mapk1 + m2_mapk1) / 2,
            "mean_mAP@5": (m1_mapk5 + m2_mapk5) / 2,
        })

    df = (
        pd.DataFrame(results)
        .sort_values(["m2_mAP@5", "m2_mAP@1"], ascending=False)
        .reset_index(drop=True)
    )

    print(df.round(4).to_string(index=False))

    best_score = df["m2_mAP@5"].max()
    best = df[df["m2_mAP@5"] == best_score]

    print(f"\nBest m2_mAP@5 = {best_score:.4f}, reached by {len(best)} distance(s): "
          f"{', '.join(best['metric'])}")

    # ----------------- Bar plot (top 10) ---------------------------
    top = df.head(10).set_index("metric")

    fig, ax = plt.subplots(figsize=(8, 7))
    top[["m1_mAP@5", "m2_mAP@5"]].plot.bar(ax=ax, width=0.8)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, rotation=90, fontsize=8)

    ax.set_title("mAP@5")
    ax.set_xlabel("")
    ax.set_ylabel("mAP@5")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(["Method 1", "Method 2"])
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    fig.suptitle("Top 10 distances")
    fig.tight_layout()
    fig.savefig("top10_distances.png", dpi=300, bbox_inches="tight")
    plt.close(fig)