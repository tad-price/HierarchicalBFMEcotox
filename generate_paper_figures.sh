#!/bin/bash
# Generate all paper figures and tables.
# Prerequisites: trained model artifacts in outputs/models/
#   python scripts/train_bfm.py
#   python scripts/generate_predictions.py

set -e

echo "=== Dataset characterization (Table 1-2, Figures 1-2) ==="
python analysis/dataset_figures.py

echo ""
echo "=== Predictive accuracy (Figures 3-4) ==="
python analysis/analyze_results.py

echo ""
echo "=== Uncertainty calibration (Figures 5-6) ==="
python analysis/uncertainty_figures.py

echo ""
echo "=== Variance decomposition (aleatoric vs epistemic) ==="
python analysis/variance_decomposition.py

echo ""
echo "=== Posterior predictive calibration curve ==="
python analysis/calibration_figures.py

echo ""
echo "=== SSD analysis: Atrazine (Figures 8-9) ==="
python analysis/ssd_analysis.py --cas 1912-24-9

echo ""
echo "=== SSD analysis: Chlorfenprop-methyl (Figures 8-9) ==="
python analysis/ssd_analysis.py --cas 14437-17-3

echo ""
echo "=== SSD MCMC ensemble: Atrazine (Figure 7) ==="
python analysis/ssd_mc_uncertainty.py --cas 1912-24-9

echo ""
echo "=== SSD MCMC ensemble: Chlorfenprop-methyl (Figure 7) ==="
python analysis/ssd_mc_uncertainty.py --cas 14437-17-3

echo ""
echo "=== HC20 for all chemicals (CSV for Figures 10-11) ==="
python analysis/ssd_mc_uncertainty.py --all-hcx

echo ""
echo "=== HC20 plots (Figures 10-11) ==="
python analysis/hcx_plots.py

echo ""
echo "=== All figures saved to outputs/figures/ ==="
