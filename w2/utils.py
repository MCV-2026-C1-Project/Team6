from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.spatial.distance as sd
from typing import Literal


# We only use .jpg because these are the only ones that contain the required images.
def load_images(folder: str) -> tuple[list[int], list[np.ndarray]]:
    """
    Load JPG images from a folder and convert them to CIE Lab.

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


def show_histogram(img: np.ndarray) -> None:
    """Display the CIELAB channel histograms.

    Args:
        img: CIELAB image as a NumPy array with three channels corresponding
            to L*, a*, and b*.

    Returns:
        None.
    """
    colors = ("black", "red", "blue")
    channel_names = ("L*", "a*", "b*")

    plt.figure(figsize=(10, 5))

    for i, (name, color) in enumerate(zip(channel_names, colors)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name)

    plt.title("CIELAB Color Histograms")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def normalize(hist: np.ndarray) -> np.ndarray:
    """
    Normalize a histogram so its values sum to 1.

    Args:
        hist: Input histogram.

    Returns:
        Normalized histogram.
    """

    return cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)


### Descriptors
def simple_descriptor(img: np.ndarray) -> np.ndarray:
    """
    Compute a normalized color histogram descriptor for an image.

    Args:
        img: Input image.

    Returns:
        NumPy array containing the normalized histograms
        of all image channels concatenated together.
    """
    hists = []
    for i in range(img.shape[2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = normalize(hist)
        hists.append(hist.flatten())

    return np.concatenate(hists)


def simple_descriptors(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute simple color histogram descriptors for multiple images.

    Args:
        images: List of input images.

    Returns:
        2D NumPy array containing one simple descriptor per input image.
    """
    return np.array([simple_descriptor(img) for img in images])


def complex_descriptor(img: np.ndarray) -> np.ndarray:
    """
    Compute a descriptor for each of the four image quadrants.

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


def complex_descriptors(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute spatial color histogram descriptors for multiple images.

    Args:
        images: List of input images.

    Returns:
        2D NumPy array containing one complex descriptor per input image.
    """
    return np.array([complex_descriptor(img) for img in images])


def color_light_block_descriptor(img: np.ndarray, grid: int = 5, ab_bins: int = 48, l_bins: int = 8) -> np.ndarray:
    """
    Compute a color + lightness histogram for each cell of a grid x grid partition.

    For each cell, a joint 2D a*b* histogram (color) and a 1D L* histogram (light)
    are L1-normalized and weighted 2/3 and 1/3, so color accounts for 2/3 of the
    information and light for 1/3. The cell descriptors are concatenated and
    divided by the number of cells, so the result sums to 1.

    Args:
        img: CIELAB image (uint8), as returned by load_images.
        grid: Number of rows and columns of the partition.
        ab_bins: Bins per color channel (a*, b*); each cell has ab_bins**2 color values.
        l_bins: Bins for the L* channel.

    Returns:
        1D descriptor of length grid**2 * (ab_bins**2 + l_bins).
    """
    rows = np.array_split(img, grid, axis=0)
    regions = [cell for row in rows for cell in np.array_split(row, grid, axis=1)]

    cell_descriptors = []
    for cell in regions:
        color = cv2.calcHist([cell], [1, 2], None, [ab_bins, ab_bins], [0, 256, 0, 256])
        light = cv2.calcHist([cell], [0], None, [l_bins], [0, 256])
        cell_descriptors.append(normalize(color).flatten() * 2/3)
        cell_descriptors.append(normalize(light).flatten() * 1 / 3)

    return np.concatenate(cell_descriptors) / grid**2

def color_light_block_descriptors(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute spatial color histogram descriptors for multiple images.

    Args:
        images: List of input images.

    Returns:
        2D NumPy array containing one complex descriptor per input image.
    """
    return np.array([color_light_block_descriptor(img) for img in images])


### Distances
def chi2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Chi-squared distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    num = (x[:, None, :] - y[None, :, :]) ** 2
    den = x[:, None, :] + y[None, :, :]
    return 0.5 * np.sum(np.divide(num, den, out=np.zeros_like(num), where=den > 0), axis=2)


def hellinger(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Hellinger distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    return np.linalg.norm(
        np.sqrt(x[:, None, :]) - np.sqrt(y[None, :, :]),
        axis=2
    ) / np.sqrt(2.0)


def bhattacharyya(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Bhattacharyya distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    coefficient = np.sum(
        np.sqrt(x[:, None, :] * y[None, :, :]),
        axis=2
    )
    return -np.log(np.clip(coefficient, np.finfo(float).tiny, 1.0))


def histogram_intersection(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise histogram intersection distance.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    return 1.0 - np.sum(
        np.minimum(x[:, None, :], y[None, :, :]),
        axis=2
    )


def total_variation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise total variation distance between descriptors.

    Args:
        x: 2D NumPy array of descriptors for the first set of images.
        y: 2D NumPy array of descriptors for the second set of images.

    Returns:
        Distance matrix of shape (len(x), len(y)).
    """
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

    """
    Compute pairwise distances between query and database descriptors.

    Args:
        query_desc: 2D NumPy array of shape (n_queries, n_features) with query descriptors as rows.
        db_desc: 2D NumPy array of shape (n_database, n_features) with database descriptors as rows.
        metric: Distance metric to use.

    Returns:
        Distance matrix of shape (n_queries, n_database).
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
# APK and MAPK are adapted from https://github.com/benhamner/Metrics.
# The ml_metrics package is no longer compatible with modern versions of setuptools, so the required functions are included here directly.
def apk(actual: list[list[int]], predicted: list[list[int]], k: int = 10) -> float:
    """
    Compute the average precision at k (AP@k) for retrieval results.

    The metric compares the ranked retrieval results against the ground-truth
    items for a query. The order of items in ``actual`` does not matter, while
    the order in ``predicted`` determines the ranking used to compute the
    average precision.

    Args:
        actual: Ground-truth BBDD IDs for each query. Each inner list contains
            the correct BBDD IDs for the corresponding query.
        predicted: Ranked BBDD IDs retrieved for each query. The order of
            elements determines their ranking.
        k: Maximum number of top-ranked predictions to consider.

    Returns:
        The average precision at k (AP@k), as a float between 0 and 1.
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


def mapk(actual: list[list[int]], predicted: list[list[int]], k: int = 10) -> float:
    """
    Compute the mean average precision at k (mAP@k) for retrieval results.

    The metric compares the ranked retrieval results against the ground-truth
    items for each query. The order of items in ``actual`` does not matter,
    while the order in ``predicted`` determines the ranking used to compute
    average precision.

    Args:
        actual: Ground-truth BBDD IDs for each query. Each inner list contains
            the correct BBDD IDs for the corresponding query.
        predicted: Ranked BBDD IDs retrieved for each query. The order of
            elements determines their ranking.
        k: Maximum number of top-ranked predictions to consider for each query.

    Returns:
        The mean average precision at k (mAP@k), as a float between 0 and 1.
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def evaluate(actual: list[list[int]], predicted: list[list[int]], k: int):
    """
    Compute the mean average precision at k (mAP@k) for retrieval results.

    The metric compares the ranked retrieval results against the ground-truth
    items for each query. The order of items in ``actual`` does not matter,
    while the order in ``predicted`` determines the ranking used to compute
    average precision.

    Args:
        actual: Ground-truth BBDD IDs for each query. Each inner list contains
            the correct BBDD IDs for the corresponding query.
        predicted: Ranked BBDD IDs retrieved for each query. The order of
            elements determines their ranking.
        k: Maximum number of top-ranked predictions to consider for each query.

    Returns:
        The mean average precision at k (mAP@k), as a float between 0 and 1.
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
def retrieve(distances: np.ndarray, db_ids: list[int], k: int = 5) -> list[list[int]]:
    """
    Return the top-k closest database image IDs for each query.

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


