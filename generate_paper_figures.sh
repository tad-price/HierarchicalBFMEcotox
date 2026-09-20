#!/bin/bash
# Regenerate every main-text figure and table from the saved model artefacts.
#
# Prerequisites:
#   python scripts/train_bfm.py --n_folds 5 --n_iter 2000 --n_burn 100
#   python scripts/generate_predictions.py --n_iter 2000 --n_burn 100
#
# Supporting Information floats are produced by scripts/rank_sweep.py and
# analysis/si_diagnostics/; see the README.

set -e

echo "=== Dataset summary, rank-frequency, replicate-SD distribution ==="
python analysis/dataset_figures.py

echo ""
echo "=== Predictive accuracy and residual diagnostics ==="
python analysis/analyze_results.py

echo ""
echo "=== Uncertainty vs data availability, example predictions ==="
python analysis/uncertainty_figures.py

echo ""
echo "=== Variance decomposition ==="
python analysis/variance_decomposition.py

echo ""
echo "=== Posterior predictive calibration ==="
python analysis/calibration_figures.py

echo ""
echo "=== Per-species SSD uncertainty ==="
python analysis/ssd_analysis.py --cas 1912-24-9
python analysis/ssd_analysis.py --cas 14437-17-3

echo ""
echo "=== Posterior ensemble SSDs ==="
python analysis/ssd_mc_uncertainty.py --cas 1912-24-9
python analysis/ssd_mc_uncertainty.py --cas 14437-17-3

echo ""
echo "=== HC20 across all chemicals, then the two HC20 figures ==="
python analysis/ssd_mc_uncertainty.py --all-hcx
python analysis/hcx_plots.py

echo ""
echo "=== Done. Outputs in outputs/figures/ ==="
