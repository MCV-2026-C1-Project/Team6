# C1. Introduction to Human and Computer Vision

| Team Members |
| -------- |
| Armengol Romero, Marçal |
| Bonet Vila, Violeta |
| Capdevila Estadella, Albert |
| Ishtay Karameh, Ali Khaled |

## Presentation

The Week 4 project presentation is available [here](https://canva.link/o0fzwaxwwr2q2xv).

## Structure

The project is organized by weeks. Each `wX` directory contains the work corresponding to **Week X**.

All project data is located in the `data/` directory. This includes the databases and datasets required for the different weeks of the project.

```text
project/
├── data/
│   ├── BBDD/
│   ├── qsd1_w1/
│   └── ...
├── w1/
│   ├── utils.py
│   ├── main.py
│   ├── test.py
│   ├── grid_search.py
│   └── results/
├── w2/
│   ├── utils.py
│   ├── main.py
│   ├── segmentation.ipynb
│   ├── gridsearches/
│   │   ├── 2d_descriptor.py
│   │   ├── 3d_descriptor.py
│   │   ├── segmentation.py
│   │   └── results/
│   └── results/
│       └── segmentation/
├── w3/
│   ├── utils.py
│   ├── main.py
│   ├── segmentation_visualization.ipynb
│   ├── searches/
│   │   ├── descriptor_search.py
│   │   ├── denoising_grid_search.py
│   │   ├── gridsearch_masking.py
│   │   └── results/
│   ├── descriptors_experiments/
│   │   ├── test_desc.py
│   │   ├── robustness_test.py
│   │   └── hue.py
│   └── results/
│       ├── QST1/
│       ├── QST2/
│       └── denoising/
├── w4/
│   ├── main.py
│   ├── denoising.py
│   ├── descriptors.py
│   ├── keypoints.py
│   ├── segmentation.py
│   ├── utils.py
│   ├── denoising.ipynb
│   ├── descriptors.ipynb
│   ├── keypoints.ipynb
│   ├── segmentation.ipynb
│   ├── searches/
│   │   ├── denoising/
│   │   │   ├── denoising_search.py
│   │   │   └── results/
│   │   ├── segmentation/
│   │   │   ├── segmentation_search.py
│   │   │   ├── segmentation_3paintings_test.py
│   │   │   └── results/
│   │   └── whole_pipeline/
│   │       ├── whole_pipeline_search.py
│   │       └── results/
│   └── visualizations/
│       └── figures/
├── README.md
└── requirements.txt
```

> `data/` must sit next to the week folders (the scripts look for it at `../data` relative to their location).

## Environment Setup

This project uses **Python 3.12.10**.

### 1. Clone the repository

```bash
git clone https://github.com/MCV-2026-C1-Project/Team6.git
cd Team6
```

### 2. Create the virtual environment

```bash
python3.12 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## How each week is organized

### `main.py` – run this to get the results

Every week has a `main.py`. Running it executes the main pipeline of that week, including loading the data, preprocessing, extracting descriptors or features, computing distances or matches, retrieving results and evaluating them.

```bash
cd wX
python main.py
```

The exact pipeline depends on the week. In the later weeks, `main.py` also contains the test stage, so the same script can be used both for validation and for generating the final test results.

### Utility modules

The utility code is progressively split into different modules as the project becomes more complex.

In Week 4, the functionality is divided into the following modules:

- `denoising.py`: functions related to image denoising and preprocessing.
- `descriptors.py`: image and painting descriptor extraction.
- `keypoints.py`: keypoint detection, description and related feature-processing methods.
- `segmentation.py`: painting/background segmentation and detection.
- `utils.py`: general-purpose functions such as image loading, preprocessing helpers, evaluation metrics, feature matching and retrieval.

These modules are imported by the main pipeline and the experimental scripts rather than being executed directly.

### `test.py` – test evaluation

In Week 1, `test.py` runs the pipeline on the test set and saves the results in the submission format as a `result.pkl` file for each method.

```bash
cd w1
python test.py
```

From Week 2 onwards, the test stage is integrated into `main.py`. Therefore, the main script can be used to evaluate the validation data and generate the corresponding test results.

## Extra scripts

Some weeks include additional scripts and notebooks for experiments, parameter searches and visualization. These are described below.

## Week 1

### `grid_search.py`

Exhaustive search over all the distance metrics available in `utils.py`, evaluating both methods with each one.

```bash
cd w1
python grid_search.py
```

It shows:

- A table with mAP@1 and mAP@5 for every distance and method, sorted from best to worst.
- The best distance(s) found.
- A bar plot of the top 10 distances (mAP@5), saved as an image in the `results/` folder (`results/grid_search_top10.png`).

## Week 2

### `gridsearches/`

Hyperparameter searches for this week's methods. Run them from `w2/gridsearches`:

```bash
cd w2/gridsearches
python 2d_descriptor.py
python 3d_descriptor.py
python segmentation.py
```

- `2d_descriptor.py`: search over the parameters of the 2D histogram descriptor, such as colour space, number of bins and block levels.
- `3d_descriptor.py`: search over the parameters of the 3D histogram descriptor.
- `segmentation.py`: search over the background removal parameters, evaluated using precision, recall, F1 and other metrics.

Results (tables and plots) are saved in `gridsearches/results/`.

### `segmentation.ipynb`

Notebook that visualizes each step of the background removal pipeline on a single QSD2 image:

1. Original image and ground-truth mask.
2. Optional prefiltering.
3. Border samples used to model the background (wall) colour.
4. HSV and Lab channels.
5. Mahalanobis distance of each pixel to the background model (2D map and 3D surface).
6. Comparison of the thresholding methods (Otsu, Triangle, GMM and wall-based threshold), with their histograms and resulting masks.
7. Post-processing: morphological closing, largest connected component and rectangle fitting.

The image and parameters can be changed in the second cell (`IMG_NAME` and `P`). By default it uses the best parameters found by `gridsearches/segmentation.py`. Figures are saved in `results/segmentation/`.

```bash
cd w2
jupyter notebook segmentation.ipynb
```

## Week 3

### `main.py`

Runs the full Week 3 pipeline on QSD2-W3: each query is denoised, up to two paintings are detected, and each painting is described with a DCT texture descriptor and retrieved from the BBDD. It prints how many paintings were detected per image and the mAP@1 and mAP@5.

With `RUN_TESTS = True`, it also runs QST1-W3 and QST2-W3 and saves the `result.pkl` files (and the QST2 masks) in `results/`. The test folders `data/qst1_w3` and `data/qst2_w3` must exist; otherwise set `RUN_TESTS = False`.

```bash
cd w3
python main.py
```

### `searches/`

Parameter searches for this week's methods:

```bash
cd w3/searches
python descriptor_search.py
python denoising_grid_search.py
python gridsearch_masking.py
```

- `descriptor_search.py`: Bayesian search (Optuna) over six texture descriptors (DCT, DWT and LBP variants) on QSD1-W3. Results are saved in `searches/results/`.
- `denoising_grid_search.py`: compares denoising filters on the noisy QSD1-W3 queries by their retrieval mAP. Results are saved in `results/denoising/`.
- `gridsearch_masking.py`: grid search over the painting detection parameters (colour space, edge threshold, kernel sizes) on QSD2-W3, evaluated with precision, recall, F1 and the number of detected paintings. Results are saved in `searches/results/gridsearch_masking/`.

### `descriptors_experiments/`

Extra experiments on the texture descriptors, all on QSD1-W3:

```bash
cd w3/descriptors_experiments
python test_desc.py
python robustness_test.py
python hue.py
```

- `test_desc.py`: mAP@1 and mAP@5 of a single descriptor, optionally with a random shift of the queries (`APPLY_SHIFT`).
- `robustness_test.py`: mAP@1 of the best configuration of each descriptor on the original, noisy, hue-shifted and shifted queries. Results are saved in `robustness_map1.csv`.
- `hue.py`: effect of a random hue shift on the DCT pyramid descriptor for different YCrCb channel combinations.

### `segmentation_visualization.ipynb`

Notebook that visualizes each step of the painting detection on a single QSD2-W3 image:

1. Prewitt gradient on the HSV S and V channels.
2. Edges thresholded at 0.5 × Otsu.
3. Closing to join the frame outlines.
4. Fill holes and opening.
5. Selection of up to two components.
6. Rectangle fitting and ordering of the paintings.
7. Evaluation on the whole dataset and comparison of variants.

The image and parameters can be changed in the parameters cell (`IDX` and `P`).

```bash
cd w3
jupyter notebook segmentation_visualization.ipynb
```

## Week 4

Week 4 extends the previous retrieval pipeline by introducing a more modular organization of the code and focusing on feature-based painting retrieval and the evaluation of different configurations.

The main pipeline is contained in `main.py`, while the functionality previously concentrated in a single utility file is now separated into dedicated modules. This makes the different stages of the pipeline easier to develop, test and reuse.

### `main.py`

`main.py` contains the complete Week 4 pipeline, including the validation and test stages.

The pipeline combines the different processing stages required for painting retrieval and evaluates the resulting matches. The test stage is also executed from `main.py`, so no separate `test.py` is required for Week 4.

```bash
cd w4
python main.py
```

### Utility modules

The Week 4 implementation separates the main computer vision components into different files:

- `denoising.py`: image denoising and preprocessing operations.
- `descriptors.py`: extraction of image descriptors used for retrieval.
- `keypoints.py`: keypoint detection and feature description.
- `segmentation.py`: detection and segmentation of paintings in the query images.
- `utils.py`: general utilities shared by the pipeline, including image loading, preprocessing helpers, evaluation metrics, feature matching and retrieval.

The general utilities also contain the evaluation functions used to compute AP and mAP, as well as the matching and retrieval functions required for multiple detected paintings. In particular, the retrieval pipeline supports both standard top-k retrieval and retrieval based on feature matching and RANSAC inliers.

### `searches/`

The `searches/` directory contains the experiments and parameter searches carried out during Week 4.

These scripts are used to compare different configurations of the pipeline and determine the parameters and methods that provide the best retrieval performance.

The results produced by the searches are stored in the corresponding results directories.

```bash
cd w4/searches
```

The individual search scripts can be run from this directory depending on the experiment being evaluated.

### `visualizations/`

The `visualizations/` directory contains the visual analysis carried out during Week 4.

These scripts are used to inspect the intermediate stages of the pipeline and visualize the behaviour of the different methods and configurations. This includes visual inspection of the processing and retrieval results and helps to analyse the effect of the parameters selected during the searches.

```bash
cd w4/visualizations
```