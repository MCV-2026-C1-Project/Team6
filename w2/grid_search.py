from utils import *
import itertools
import pickle
import pandas as pd


# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BBDD_DIR = DATA_DIR / "BBDD"
QSD1 = DATA_DIR / "qsd1_w2"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Parameter grid for color_light_block_descriptor
GRIDS = [2, 3, 4, 5, 6]
AB_BINS = [8,24, 48, 96, 128]
L_BINS = [4, 8, 16]
METRICS = ["histogram_intersection", "hellinger", "chi2", "bhattacharyya", "total_variation", "jensenshannon"]


def get_distances_by_query(query_desc: np.ndarray, db_desc: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute get_distances one query at a time.

    The custom distances broadcast to (n_queries, n_database, n_features), which does not
    fit in memory for the largest descriptors, so the queries are processed one by one.

    Args:
        query_desc: 2D NumPy array of shape (n_queries, n_features).
        db_desc: 2D NumPy array of shape (n_database, n_features).
        metric: Distance metric to use.

    Returns:
        Distance matrix of shape (n_queries, n_database).
    """
    return np.vstack([get_distances(q[None, :], db_desc, metric=metric) for q in query_desc])


if __name__ == "__main__":
    # Load database, query images, and ground-truth correspondences.
    db_ids, db_images = load_images(BBDD_DIR)
    query_ids, query_images = load_images(QSD1)

    with open(QSD1 / "gt_corresps.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    results = []
    combinations = list(itertools.product(GRIDS, AB_BINS, L_BINS))

    # Evaluate every (grid, ab_bins, l_bins) combination with each distance metric.
    for i, (grid, ab_bins, l_bins) in enumerate(combinations, start=1):
        print(f"[{i}/{len(combinations)}] grid={grid}, ab_bins={ab_bins}, l_bins={l_bins}")

        db_desc = np.array([color_light_block_descriptor(img, grid, ab_bins, l_bins) for img in db_images])
        query_desc = np.array([color_light_block_descriptor(img, grid, ab_bins, l_bins) for img in query_images])

        for metric in METRICS:
            try:
                dists = get_distances_by_query(query_desc, db_desc, metric)
                topk5 = retrieve(dists, db_ids, k=5)
                mapk1 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=1)
                mapk5 = evaluate(ground_truth, dict(zip(query_ids, topk5)), k=5)

            except Exception as e:
                print(f"Skipping {metric}: {e}")
                continue

            results.append(
                {
                    "grid": grid,
                    "ab_bins": ab_bins,
                    "l_bins": l_bins,
                    "metric": metric,
                    "dims": db_desc.shape[1],
                    "mAP@1": mapk1,
                    "mAP@5": mapk5,
                }
            )

    # Rank configurations by mAP@5 (ties broken by mAP@1).
    df = (
        pd.DataFrame(results)
        .sort_values(
            ["mAP@5", "mAP@1"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "grid_search.csv", index=False)

    print(df.head(20).round(4).to_string(index=False))

    # Report the best-performing configuration(s).
    best_score = df["mAP@5"].max()
    best = df[df["mAP@5"] == best_score]

    print(f"\nBest mAP@5 = {best_score:.4f}, reached by {len(best)} configuration(s):")
    print(best.round(4).to_string(index=False))

    # Plot mAP@1 and mAP@5 for the top 10 configurations.
    top = df.head(10).copy()
    top["config"] = top.apply(
        lambda r: f"g={r.grid}, ab={r.ab_bins}, l={r.l_bins}\n{r.metric}",
        axis=1,
    )
    top = top.set_index("config")

    fig, ax = plt.subplots(figsize=(10, 7))
    top[["mAP@1", "mAP@5"]].plot.bar(
        ax=ax,
        width=0.8,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            rotation=90,
            fontsize=8,
        )

    ax.set_title("color_light_block_descriptor")
    ax.set_xlabel("")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    fig.suptitle("Top 10 configurations")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "grid_search_top10.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
