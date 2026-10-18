# C1. Introduction to Human and Computer Vision

| Team Members     |
| -------- |
| Armengol Romero, Marçal |
| Bonet Vila, Violeta |
| Capdevila Estadella, Albert |
| Ishtay Karameh , Ali Khaled |

## Structure

The project is organized by weeks. Each `wX` directory contains the work corresponding to **Week X**.
All project data is located in the `data/` directory. This includes the databases and datasets required for the different weeks of the project.

```
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
│   ├── main.py             # validation + test in a single script
│   ├── gridsearches/
│   │   ├── 2d_descriptor.py
│   │   ├── 3d_descriptor.py
│   │   ├── segmentation.py
│   │   └── results/
│   └── results/
└── ...
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

Every week has a `main.py`. Running it executes the full pipeline of that week (loading data, extracting descriptors, computing distances, retrieving results) and prints the evaluation results (mAP@K) in the terminal.

```bash
cd wX
python main.py
```

### `utils.py` – shared functions

Every week has a `utils.py` that always contains all the functions used by that week's scripts (image loading, descriptors, distances, retrieval and evaluation metrics). It is not run directly: it is imported by the other scripts with `from utils import *`.

### `test.py` – run this to get the test results (Week 1 only)

In Week 1, `test.py` runs the same pipeline on the test set and saves the results in the submission format as a `result.pkl` file for each method.

```bash
cd w1
python test.py
```

From Week 2 onwards, the test step is integrated into `main.py`: a single run evaluates on the validation set and generates the `result.pkl` files for the test set.

### Extra scripts

Some weeks include additional scripts for experiments. Each one is described below.

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

Hyperparameter searches for this week's methods. Run them from the `w2/gridsearches`:

```bash
cd w2/gridsearches
python 2d_descriptor.py
python 3d_descriptor.py
python segmentation.py
```

- `2d_descriptor.py`: search over the parameters of the 2D histogram descriptor (e.g. colour space, number of bins, block levels).
- `3d_descriptor.py`: same for the 3D histogram descriptor.
- `segmentation.py`: search over the background removal parameters, evaluated with precision, recall and F1 among others.

Results (tables and plots) are saved in `gridsearches/results/`.