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
        .sort_values(["mean_mAP@5", "mean_mAP@1"], ascending=False)
        .reset_index(drop=True)
    )

    print(df.round(4).to_string(index=False))
    print(f"\nBest shared distance: {df.loc[0, 'metric']}")