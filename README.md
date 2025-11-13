# Stratified Active Learning

This repository contains the implementation of stratified active learning methods for multi-label bioacoustic classification, with a focus on efficient sample selection strategies across spatial and temporal strata.

Pre-print available [here](https://doi.org/10.1101/2025.09.01.673472)


### Citation

If you use this code in your research, please cite:

```bibtex
@article{mcewen2025stratified,
  title={Stratified Active Learning for Spatiotemporal Generalisation in Large-Scale Bioacoustic Monitoring},
  author={McEwen, Ben and Bernard, Corentin and Stowell, Dan},
  journal={bioRxiv},
  pages={2025--09},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Key Features

- **Multiple Stratification Methods**: Spatial (location-based), temporal (time-based), and species-based stratification
- **Uncertainty Quantification**: Binary entropy, ratio max, and cluster-based uncertainty measures
- **Active Learning Pipeline**: Complete implementation with batch sampling and accumulation strategies
- **Evaluation Framework**: Per-strata performance evaluation using mAP, cmAP, and F1 scores

## Installation

1. Clone the repository:
```bash
https://github.com/BenMcEwen1/Stratified-Sampling.git
cd Stratified-Sampling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.2.1
- NumPy 1.26.3
- Pandas 2.2.2
- scikit-learn (via scipy)
- Matplotlib 3.9.2
- Seaborn 0.13.2
- tqdm 4.66.3

## Core Modules

### `sampling.py` - Uncertainty Quantification

The main module implementing various sampling strategies:

**Class: `UncertaintyQuantification`**

Key methods:
- `binaryEntropy()`: Computes maximum binary entropy across classes for uncertainty estimation
- `ratioMax()`: Ratio-based uncertainty quantification
- `clusterEntropy()`: Hierarchical agglomerative clustering with entropy-based selection
- `stratified()`: Stratified sampling across predefined strata with optional weighting
- `resample()`: Core resampling method supporting multiple uncertainty strategies

**Usage Example:**
```python
from sampling import UncertaintyQuantification

# Initialize with embeddings, labels, and desired sample count
sampler = UncertaintyQuantification(x=embeddings, y=labels, samples_num=100)

# Stratified sampling with binary entropy
strata_indices = sampler.stratified(
    model=classifier_model,
    sorted_indices=strata_dict,
    method="binary",
    weights=None
)
```

### `adapter.py` - Dataset Loader

Middleware for loading different bioacoustic datasets with a unified interface.

**Supported Datasets:**
- **WABAD**: World Annotated Bird Acoustic Dataset
- **AnuraSet**: Anuran (frog) call classification dataset

**Function: `loader(dataset_name, sub_directory, dataset_path=None)`**

Parameters:
- `dataset_name` (str): Name of dataset - `"wabad"` or `"anuraset"`
- `sub_directory` (str): Data split - `"train"`, `"test"`, or `"validation"`
- `dataset_path` (str, optional): Custom path to dataset root directory

Returns:
- `embeddings` (torch.Tensor): Feature embeddings of shape (N, embedding_dim)
- `labels` (torch.Tensor): Multi-label binary matrix of shape (N, num_classes)
- `filenames` (list): List of filenames corresponding to each sample

**Usage Examples:**

```python
from adapter import loader

# Load AnuraSet training data
embeddings, labels, filenames = loader("anuraset", sub_directory="train")
# Returns: torch.Size([62191, 1024]), torch.Size([62191, 42])

# Load WABAD test data
embeddings, labels, filenames = loader("wabad", sub_directory="test")
# Returns: torch.Size([16123, 1024]), torch.Size([16123, 106])

# Load WABAD validation data
embeddings, labels, filenames = loader("wabad", sub_directory="validation")

# Use custom dataset path
embeddings, labels, filenames = loader(
    "anuraset",
    sub_directory="train",
    dataset_path="/path/to/anuraset"
)
```

**Dataset Structure Requirements:**

For **AnuraSet**:
```
anuraset/
├── train/
│   ├── embeddings.pt
│   ├── labels.pt
│   └── data.csv
└── test/
    ├── embeddings.pt
    ├── labels.pt
    └── data.csv
```

For **WABAD**:
```
WABAD/
└── data_files/
    ├── embeddings_train.pkl
    ├── embeddings_test.pkl
    ├── embeddings_validation.pkl
    ├── dataframe_train.pkl
    ├── dataframe_test.pkl
    ├── dataframe_validation.pkl
    └── BirdNET_GLOBAL_6K_V2.4_Labels.txt
```

## Notebooks

### `spatiotemporal_stratified.ipynb`

Comprehensive experiments comparing different stratification strategies with spatial and temporal dimensions.

**Quick Start:**

1. Select your dataset at the top of the notebook:
```python
# Dataset selection
DATASET = "wabad"  # Options: "anuraset" or "wabad"

# Load data using adapter
from adapter import loader
tensor_x_train, tensor_y_train, train_filenames = loader(DATASET, sub_directory="train")
tensor_x_test, tensor_y_test, test_filenames = loader(DATASET, sub_directory="test")
```

2. Configure active learning hyperparameters:
```python
n_samples = 54              # Samples per round
initial_samples = 20        # Initial labeled samples
initial_sampling_mode = "stratified"  # or "random"
sampling_mode = "stratified"          # or "random", "cluster"
sub_sampling_mode = "binary"          # or "ratio_max"
strata_mode = "spatial"               # or "temporal", "custom"
```

**Stratification Modes:**
- **Spatial**: Stratify by recording location (26 locations in WABAD, varies in AnuraSet)
- **Temporal**: Stratify by time period (year, month, or custom periods)
- **Custom**: User-defined temporal groupings for seasonal analysis

**Key Features:**
- Stratified regularization loss to minimize variance across strata
- Per-strata performance evaluation and visualization
- JS-divergence analysis for measuring distribution differences between strata
- Mutual information calculation between strata and labels
- Weighted sampling based on stratum similarity

**Stratification Helper Functions:**

```python
# Spatial stratification by location
sorted_indices_spatial = spatial_split(train_filenames)
# Returns: {'LOC001': [0, 15, 23, ...], 'LOC002': [...], ...}

# Temporal stratification by year/month
sorted_indices_temporal = temporal_split(train_filenames, res=6)  # YYYYMM
# Returns: {'201909': [...], '201910': [...], ...}

# Custom temporal periods
sorted_indices_custom = temporal_custom(train_filenames)
# Returns: {'A': [...], 'B': [...], 'C': [...]}
```

### Uncertainty Measures

**Binary Entropy** (primary method):
```python
entropy = -(p * log(p) + (1-p) * log(1-p))
uncertainty = max(entropy across classes)
```

**Ratio Max**:
```python
uncertainty = (0.5 - |p - 0.5|) / (0.5 + |p - 0.5|)
```

**Cluster-based**: Hierarchical clustering with within-cluster entropy maximization

## Datasets

This project supports two bioacoustic datasets through the unified `adapter.py` loader.

### WABAD (World Annotated Bird Acoustic Dataset)

Bird species classification dataset with rich spatial and temporal metadata.

**Dataset Statistics:**
- **Training samples**: 13,453 embeddings (1024-dim BirdNet embeddings)
- **Validation samples**: Available
- **Test samples**: 16,123 embeddings
- **Species**: 106 bird species (filtered from 6,000+ BirdNet classes)
- **Locations**: 26 recording sites across various habitats
- **Temporal span**: Multiple years (2016-2023)

**Expected Directory Structure:**
```
WABAD/
└── data_files/
    ├── embeddings_train.pkl
    ├── embeddings_test.pkl
    ├── embeddings_validation.pkl
    ├── dataframe_train.pkl
    ├── dataframe_test.pkl
    ├── dataframe_validation.pkl
    └── BirdNET_GLOBAL_6K_V2.4_Labels.txt
```

### AnuraSet

Anuran (frog) call classification dataset for amphibian bioacoustic monitoring.

**Dataset Statistics:**
- **Training samples**: 62,191 embeddings (1024-dim)
- **Test samples**: 31,187 embeddings
- **Species**: 42 anuran species
- **Locations**: Multiple recording sites
- **Audio type**: Frog calls and vocalizations


**Expected Directory Structure:**
```
anuraset/
├── train/
│   ├── embeddings.pt      # Torch tensor of embeddings
│   ├── labels.pt          # Torch tensor of labels
│   └── data.csv           # Metadata with 'fname' column
└── test/
    ├── embeddings.pt
    ├── labels.pt
    └── data.csv
```
