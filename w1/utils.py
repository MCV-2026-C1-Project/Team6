from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance as sd
from typing import Literal

#We only use .jpg because these are the only ones that contain the required images.
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

# Function adapted to CieLab
def load_images_lab(folder: str) -> tuple[list[int], list[np.ndarray]]:
    """Load JPG images from a folder and convert them to CIE Lab.

    Args:
        folder: Path to the folder containing the images.

    Returns:
        Tuple (ids, images) where ids[i] is the ID of images[i].
        Images are stored in Lab color space.
    """

    ids: list[int] = []
    images: list[np.ndarray] = []

    for path in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(path))

        if img is None:
            print(f"Could not read {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        image_id = int(path.stem.split("_")[-1])

        ids.append(image_id)
        images.append(img)

    return ids, images


def show_histogram(img):
    """Given an assumed BGR image plot its histogram and the hole image"""
    colors = ('b','g', 'r')
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
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
    

def normalize(hist: np.ndarray) -> np.ndarray:
    """Normalize a histogram so its values sum to 1.

    Args:
        hist: Input histogram.

    Returns:
        Normalized histogram.
    """

    return cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)


### Descriptors
# Added: normalize the histogram

def simple_descriptor(img):
    """Given an image recieve the 3 1d colour histograms compressed in a single numpy array"""
    hists = []
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = normalize(hist)
        hists.append(hist.flatten())

    return np.concatenate(hists)


def simple_descriptors(images):
    return np.array([simple_descriptor(img) for img in images])


def complex_descriptor(img: np.ndarray) -> np.ndarray:
    """Compute a descriptor for each of the four image quadrants.

    Args:
        img: Input image.
    
    Returns:
        Concatenated descriptors for the four quadrants.
    """

    rows = np.array_split(img, 3, axis = 0)
    regions = [cell for row in rows for cell in np.array_split(row, 3, axis = 1)]

    return np.concatenate([
        simple_descriptor(subimage)
        for subimage in regions
    ])


def complex_descriptors(images):
    return np.array([complex_descriptor(img) for img in images])


### Distances

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

DISTANCES = ["bhattacharyya", 'braycurtis', 'canberra', 'chebyshev', "chi2", 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', "hellinger", "histogram_intersection", 'jaccard', 'jensenshannon', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', "total_variation", 'yule']

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

### Evaluation
# apk and mapk copied from https://github.com/benhamner/Metrics
# The ml_metrics package no longer
# installs with modern setuptools, so the functions are included here.
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def evaluate(actual, predicted, k):
    """Compute mAP@k of the retrieval results against the ground truth.

    Args:
        actual: Ground truth as loaded from gt_corresps.pkl, a list of lists
            where actual[query_id] holds the correct BBDD ids for that query.
        predicted: Ranked BBDD ids for each query, either a dict
            {query_id: [ids]} or a list of lists in the same order as actual.
        k: Number of top results to consider.

    Returns:
        mAP@k as a float between 0 and 1.
    """

    # The query folder can have fewer images than the ground truth
    # (qsd1_w1 has 24 images for 30 entries), so match them by query id
    if isinstance(predicted, dict):
        query_ids = sorted(predicted)
        actual = [actual[qid] for qid in query_ids]
        predicted = [list(predicted[qid]) for qid in query_ids]
    elif len(actual) != len(predicted):
        raise ValueError(
            f'Got {len(predicted)} predictions for {len(actual)} ground truth '
            'entries. Pass predictions as a dict {query_id: [ids]}.'
        )

    return mapk(actual, predicted, k)

### Retrieval

def retrieve(distances, db_ids, k=5):
    """Return the top-k closest database image IDs for each query.

    Args:
        distances: Distance matrix of shape (n_queries, n_database).
            Lower values are considered better matches.
        db_ids: Database image IDs corresponding to the columns
            of the distance matrix.
        k: Number of database images to retrieve per query.

    Returns:
        List of lists containing the top-k database image IDs
        for each query, ordered from smallest to largest distance.
    """

    if distances.ndim != 2:
        raise ValueError("distances must be a 2D array")

    if distances.shape[1] != len(db_ids):
        raise ValueError(
            "Number of database IDs must match the number of distance columns"
        )

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > len(db_ids):
        raise ValueError("k cannot be larger than the database")

    sorted_indices = np.argsort(distances, axis=1)
    top_k_indices = sorted_indices[:, :k]

    results = []

    for query_indices in top_k_indices:
        query_results = [db_ids[i] for i in query_indices]
        results.append(query_results)

    return results
