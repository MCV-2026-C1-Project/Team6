from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance as sd
from typing import Literal

#We only use .jpg because these are the only ones that contain the required images. Ask a teacher what .png files are for. 
def load_images(folder):
    """Read every .jpg in folder and return {image_id: BGR image}."""
    images = {}
    for path in sorted(Path(folder).glob('*.jpg')):
        img = cv2.imread(str(path))
        if img is None:
            print(f'Could not read {path}')
            continue
        image_id = int(path.stem.split('_')[-1])
        images[image_id] = img
    return images

def show_histogram(img):
    """Given an assumed BGR image plot its histogram and the hole image"""
    colors = ('b','g', 'r')
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.set_title("Original image")
    ax_img.axis('off')

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax_hist.plot(hist, color=col)
        ax_hist.set_xlim([0, 256])

    ax_hist.set_title("RGB Color Histogram")
    ax_hist.set_xlabel("Pixel Intensity")
    ax_hist.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

#This will later need normalization. 
def compute_histogram(img):
    """Given an image recieve the 3 1d colour histograms compressed in a single numpy array"""
    hists = []
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hists.append(hist.flatten())

    return np.concatenate(hists)


# Distances

def chi2(x, y):
    num = (x[:, None, :] - y[None, :, :]) ** 2
    den = x[:, None, :] + y[None, :, :]
    return 0.5 * np.sum(np.divide(num, den, out=np.zeros_like(num), where=den > 0), axis=2)


def hellinger(x, y):
    return np.linalg.norm(
        np.sqrt(x[:, None, :]) - np.sqrt(y[None, :, :]),
        axis=2
    ) / np.sqrt(2.0)


def bhattacharyya(x, y):
    coefficient = np.sum(
        np.sqrt(x[:, None, :] * y[None, :, :]),
        axis=2
    )
    return -np.log(np.clip(coefficient, np.finfo(float).tiny, 1.0))


def histogram_intersection(x, y):
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    return 1.0 - np.sum(
        np.minimum(x[:, None, :], y[None, :, :]),
        axis=2
    )

def total_variation(x, y):
    return 0.5 * np.sum(
        np.abs(x[:, None, :] - y[None, :, :]),
        axis=2
    )


def get_distances(
        query_desc: np.ndarray,
        db_desc: np.ndarray,
        metric: Literal[
            "bhattacharyya", 'braycurtis', 'canberra', 'chebyshev', "chi2", 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', "hellinger", "histogram_intersection", 'jaccard', 'jensenshannon', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', "total_variation", 'yule'
        ] = 'euclidean'
    ) -> np.ndarray:

    """Compute pairwise distances between query and database descriptors.

    Args:
        query_desc (np.ndarray): 2D array of shape (n_queries, n_features)
            with query descriptors as rows.
        db_desc (np.ndarray): 2D array of shape (n_database, n_features)
            with database descriptors as rows.
        metric (str): Distance metric to use.

    Returns:
        np.ndarray: Distance matrix of shape (n_queries, n_database).
    """

    other_metrics = {
        "chi2": chi2,
        "hellinger": hellinger,
        "bhattacharyya": bhattacharyya,
        "histogram_intersection": histogram_intersection,
        "total_variation": total_variation,
    }

    if metric in other_metrics:
        return other_metrics[metric](query_desc, db_desc)

    return sd.cdist(query_desc, db_desc, metric)