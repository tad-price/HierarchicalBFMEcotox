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

- Python 3.10+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- joblib
- pyarrow (for parquet support)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Place the following files in `data/raw/`:

1. `ecotox_mortality_processed.csv` - Ecotoxicology mortality data
2. `ecotox_properties_with-oecd-function.csv` - Chemical properties

These files are not included in the repository due to size. Contact the authors for data access.

## Usage

### 1. Train the Model (Cross-Validation)

Trains the hierarchical BFM with 3-fold cross-validation and saves out-of-fold predictions:

```bash
python scripts/train_bfm.py
```

**Outputs to `outputs/models/`:**
- `oof_mean.npy` - Mean predictions
- `oof_epistemic.npy` - Epistemic variance
- `oof_aleatoric.npy` - Aleatoric variance

### 2. Generate Full Predictions

Trains on full dataset and generates predictions for all (chemical, species, duration) combinations:

```bash
python scripts/generate_predictions.py
```

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

## Pre-computed Outputs

If you have access to pre-computed model outputs, place them in `outputs/models/`:

- `oof_mean.npy`
- `oof_epistemic.npy`
- `oof_aleatoric.npy`
- `full_predictions.parquet`
- `trained_model.pkl` (optional)

This allows running analysis scripts without retraining.

## Model Details

The Hierarchical BFM uses Gibbs sampling with:
- **Latent dimensionality**: k=32
- **Iterations**: 200 (100 burn-in)
- **Per-chemical precision**: α_c ~ Gamma(a₀, b₀) with a₀=b₀=1

Key features:
- Categorical features: species, CAS (chemical), duration, taxonomic family, taxonomic class
- Numerical features: log molecular weight, cLogP
- Target: log mg/L concentration (centered)

## Citation

If you use this code, please cite:

```bibtex
@article{hierarchical-bfm-2025,
  title={Hierarchical Bayesian Factorization Machines for Ecotoxicology Prediction with Uncertainty Quantification},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
