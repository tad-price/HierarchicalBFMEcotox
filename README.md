# Hierarchical Bayesian Factorization Machine for Ecotoxicology

This repository contains the code for training and analyzing a Hierarchical Bayesian Factorization Machine (BFM) for predicting chemical toxicity with uncertainty quantification.

## Overview

The Hierarchical BFM extends standard Bayesian Factorization Machines by learning a separate noise precision parameter (α) for each chemical, enabling:

- **Aleatoric uncertainty**: Per-chemical irreducible measurement noise (1/α)
- **Epistemic uncertainty**: Model uncertainty from posterior sampling

This decomposition allows for more accurate Species Sensitivity Distributions (SSDs) with proper uncertainty quantification.

## Repository Structure

```
hierarchical-bfm-paper/
├── src/                          # Source code
│   ├── model/
│   │   └── hierarchical_bfm.py   # HierarchicalBFM class
│   └── data/
│       └── load_ecotox.py        # Data loading utilities
│
├── scripts/                      # Training scripts
│   ├── train_bfm.py              # Train with cross-validation (OOF predictions)
│   └── generate_predictions.py   # Generate full prediction matrix
│
├── analysis/                     # Analysis scripts
│   ├── analyze_results.py        # Model performance & uncertainty analysis
│   ├── ssd_analysis.py           # Species Sensitivity Distribution plots
│   └── compare_hc5.py            # HC5 comparison (traditional vs BFM)
│
├── data/
│   └── raw/                      # Raw data files (see Data Setup)
│
├── outputs/
│   ├── models/                   # Saved models and predictions
│   └── figures/                  # Generated plots
│
└── notebooks/                    # Optional analysis notebooks
```

## Installation

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Place the following files in `data/raw/`:

1. `ecotox_mortality_processed.csv` - Ecotoxicology mortality data
2. `ecotox_properties_with-oecd-function.csv` - Chemical properties

The data comes from the authors of the ADORE dataset, and the files can be found here: https://gitlab.renkulab.io/mltox/adore , specifically in data/processed for the mortality data, and in the chemicals folder for the properties data. 


## Usage

### 1. Train the Model (Cross-Validation)

Trains the hierarchical BFM with k-fold cross-validation and saves out-of-fold predictions:

```bash
python scripts/train_bfm.py
```
It supports the following CLI args:

- `--n_folds`: Number of cross-validation folds (default: 3)
- `--n_iter`: Number of Gibbs sampling iterations (default: 200)
- `--n_burn`: Number of burn-in iterations (default: 100)

**Outputs to `outputs/models/`:**
- `oof_mean.npy` - Mean predictions
- `oof_epistemic.npy` - Epistemic variance
- `oof_aleatoric.npy` - Aleatoric variance

### 2. Generate Full Predictions

Trains on full dataset and generates predictions for all (chemical, species, duration) combinations:

```bash
python scripts/generate_predictions.py
```
Supports --n_iter and --n_burn in cli.

**Outputs to `outputs/models/`:**
- `trained_model.pkl` - Trained model (~130MB)
- `full_predictions.parquet` - Full prediction matrix (~560MB)

### 3. Run Analysis

After training, run the analysis scripts:

```bash
# Model performance and uncertainty analysis
python analysis/analyze_results.py

# Species Sensitivity Distribution plots
python analysis/ssd_analysis.py

# HC5 comparison across all chemicals
python analysis/compare_hc5.py
```

**Outputs to `outputs/figures/`:**
- `predicted_vs_measured_48h.png`
- `bias_analysis_48h.png`
- `uncertainty_exploration.png`
- `aleatoric_calibration.png`
- `ssd_*.png` - Various SSD plots
- `hc5_correlation_48h.png`
- `hc5_comparison_48h.csv`

Some non-image data is also output in the terminal directly. 

## Model Details

The Hierarchical BFM uses Gibbs sampling with:
- **Latent dimensionality**: k=32
- **Iterations**: 200 (100 burn-in) [in the paper 2000 iterations with 50 burn-in is used, depending on hardware this can take multiple hours to run]
- **Per-chemical precision**: α_c ~ Gamma(a₀, b₀) with a₀=b₀=1

Key features:
- Categorical features: species, CAS (chemical), duration, taxonomic family, taxonomic class
- Numerical features: log molecular weight, cLogP
- Target: log mg/L concentration (centered)
- 
## LLMs disclaimer
Claude code was used to assist creating this repo, especially in creating the code to generate the figures, and in improving the readability of the core code. 

## Citation

If you use this code, please cite:

## License

MIT License - see LICENSE file for details.
