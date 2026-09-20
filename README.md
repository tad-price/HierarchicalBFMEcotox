# Hierarchical Bayesian Factorization Machine for Ecotoxicology

Code for **Uncertainty Quantification for Aquatic Ecotoxicity Prediction**: a
Bayesian Factorization Machine that predicts LC50 values for untested
species–chemical pairs and reports, for every prediction, how far it can be
trusted.

## Overview

The model extends the Bayesian Factorization Machine of Rendle et al. by
replacing the single global noise precision with one precision per chemical,
drawn from a shared learned prior. Each prediction then carries two
uncertainties:

- **Epistemic** — the spread across posterior samples, which shrinks as more
  data constrains the fit.
- **Aleatoric** — the per-chemical noise variance (1/α_c), which does not.

Both are propagated into Species Sensitivity Distributions and the hazardous
concentrations derived from them.

## Repository structure

```
├── src/
│   ├── model/hierarchical_bfm.py   # HierarchicalBFM: Gibbs sampler
│   └── data/load_ecotox.py         # dataset loading and feature construction
│
├── scripts/
│   ├── train_bfm.py                # cross-validated fit -> out-of-fold arrays
│   ├── generate_predictions.py     # full-data fit -> all-triplet predictions
│   └── rank_sweep.py               # refits across factorization ranks (SI)
│
├── analysis/
│   ├── dataset_figures.py          # dataset summary, rank-frequency, replicate SDs
│   ├── analyze_results.py          # accuracy and residual diagnostics
│   ├── uncertainty_figures.py      # uncertainty vs data, example predictions
│   ├── variance_decomposition.py   # aleatoric/epistemic split by observation count
│   ├── calibration_figures.py      # posterior predictive calibration
│   ├── ssd_analysis.py             # per-species SSD uncertainty
│   ├── ssd_mc_uncertainty.py       # posterior ensemble SSDs and HCx
│   ├── hcx_plots.py                # HC20 forest and correlation plots
│   └── si_diagnostics/             # numerical claims made in the SI
│
├── notebooks/reproduce_paper.ipynb # every main-text float, in manuscript order
├── generate_paper_figures.sh       # the same, as a shell pipeline
│
├── data/raw/                       # input data (not versioned, see below)
└── outputs/
    ├── models/                     # fitted artefacts (not versioned)
    └── figures/                    # figures and tables
```

## Installation

```bash
pip install -r requirements.txt
```

## Data setup

Place these two files in `data/raw/`:

1. `ecotox_mortality_processed.csv`
2. `ecotox_properties_with-oecd-function.csv`

Both come from the ADORE dataset: https://gitlab.renkulab.io/mltox/adore —
the mortality data is in `data/processed`, the properties data in the
`chemicals` folder.

## Reproducing the paper

### 1. Fit the model

The cross-validated run produces the out-of-fold predictions behind every
accuracy and uncertainty result:

```bash
python scripts/train_bfm.py --n_folds 5 --n_iter 2000 --n_burn 100
```

Table 2 also reports a second run grouped by chemical–species pair rather than
by triplet, which is the like-for-like comparison with Posthuma et al.:

```bash
python scripts/train_bfm.py --n_folds 5 --n_iter 2000 --n_burn 100 \
    --group_key pair --out_dir outputs/models_pairCV
```

The SSDs and hazardous concentrations come instead from a single fit to all
observations:

```bash
python scripts/generate_predictions.py --n_iter 2000 --n_burn 100
```

Defaults are smaller than the published settings (3 folds, 200 iterations) so
that a first run finishes quickly; the paper uses the values above, which take
several hours.

### 2. Regenerate the figures and tables

Either open `notebooks/reproduce_paper.ipynb`, which runs every main-text float
in manuscript order, or:

```bash
bash generate_paper_figures.sh
```

### Where each float comes from

| Manuscript float | Produced by | Output |
|---|---|---|
| Table 1 — dataset summary | `dataset_figures.dataset_summary_table` | `dataset_summary.csv` |
| Figure 1 — rank-frequency | `dataset_figures.plot_rank_frequency` | `rank_freq.png` |
| Figure 2 — replicate SDs | `dataset_figures.plot_replicate_sd_distribution` | `replicate_sd_distribution_all_durations.png` |
| Table 2 — CV grouping schemes | `train_bfm.py`, both `--group_key` values | printed |
| Figure 3 — predicted vs measured | `analyze_results.plot_predicted_vs_measured_correlation` | `predicted_vs_measured_48h.png` |
| Figure 4 — residual diagnostics | `analyze_results.analyze_prediction_bias` | `bias_analysis_48h.png` |
| Table 3 — replicate calibration | `dataset_figures.replicate_calibration_table` | `replicate_calibration.csv` |
| Figure 5 — uncertainty vs data | `uncertainty_figures.plot_uncertainty_vs_observations` | `uncertainty_vs_observations_48h.png` |
| Figure 6 — variance composition | `variance_decomposition.py` | `variance_composition_vs_nobs.png` |
| Figure 7 — calibration curve | `calibration_figures.py` | `calibration_curve.png`, `.csv` |
| Figure 8 — example predictions | `uncertainty_figures.plot_example_predictions` | `example_predictions_48h.png` |
| Figure 9 — per-species SSD uncertainty | `ssd_analysis.py --cas ...` | `*_novel_uncertainty.png` |
| Figure 10 — ensemble SSD | `ssd_mc_uncertainty.py --cas ...` | `ssd_uncertainty_*_48h.png` |
| Figure 11 — HC20 forest | `hcx_plots.plot_hc_forest` | `hc20_forest_plot.png` |
| Figure 12 — HC20 correlation | `hcx_plots.plot_hc_correlation` | `hc20_correlation_trad_vs_mc.png` |
| SI rank sensitivity | `scripts/rank_sweep.py` | `outputs/rank_sweep/` |
| SI aleatoric vs replicates | `si_diagnostics/aleatoric_vs_replicates.py` | printed, `ale_vs_within.csv` |
| SI HC20 decomposition, ESS | `si_diagnostics/ssd_species_selection.py` | printed, `ssd_species_selection.csv` |
| SI null calibration ratio | `si_diagnostics/null_calibration.py` | printed, `table2_null_calibration.csv` |

Figures are written to `outputs/figures/` unless noted. Several scripts also
print numbers that appear in the text but in no figure.

## Model details

- Factorization rank k = 32
- Gibbs sampling, 2000 iterations with 100 burn-in → 1,900 posterior samples
- Per-chemical precision α_c ~ Gamma(a₀, b₀); a₀ set by an empirical-Bayes
  pre-pass, b₀ learned under a Gamma hyperprior
- Categorical features: species, chemical (CAS), duration, taxonomic family,
  taxonomic class
- Numerical features: log molecular weight, cLogP
- Target: log₁₀ LC50 in mg/L, centered

The full specification, including every conditional posterior, is in the
paper's Supporting Information.

## LLM disclaimer

Claude Code was used to assist in creating this repository, especially in
writing the figure-generation code and in improving the readability of the
core model code.

## Citation

If you use this code, please cite:

## License

MIT License — see LICENSE.
