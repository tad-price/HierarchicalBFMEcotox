#!/usr/bin/env bash
#
# generate_paper_figures.sh - Reproduce all figures and tables from the paper.
#
# Prerequisites:
#   1. Install dependencies:  pip install -r requirements.txt
#   2. Place ADORE data in data/raw/ (see data/raw/README.md)
#   3. Train the model:       python scripts/train_bfm.py
#   4. Generate predictions:  python scripts/generate_predictions.py
#
# Then run this script to produce every figure and table.
#
# Usage:
#   bash generate_paper_figures.sh
#
set -euo pipefail

echo "============================================================"
echo "  Generating all paper figures and tables"
echo "============================================================"
echo ""

# --- Dataset characterization (Table 1, Table 2, Figures 1-2) ---
echo ">>> Dataset figures (Table 1, Table 2, Figures 1-2)"
python analysis/dataset_figures.py
echo ""

# --- Model performance (Figures 3-4) ---
echo ">>> Model performance (Figures 3-4)"
python analysis/analyze_results.py
echo ""

# --- Uncertainty calibration (Figures 5-6) ---
echo ">>> Uncertainty calibration (Figures 5-6)"
python analysis/uncertainty_figures.py
echo ""

# --- SSD plots for Atrazine (Figures 7a, 8a, 9a) ---
echo ">>> SSD: Atrazine (Figures 7a, 8a, 9a)"
python analysis/ssd_analysis.py --cas 1912-24-9
python analysis/ssd_mc_uncertainty.py --cas 1912-24-9
echo ""

# --- SSD plots for Chlorfenprop-methyl (Figures 7b, 8b, 9b) ---
echo ">>> SSD: Chlorfenprop-methyl (Figures 7b, 8b, 9b)"
python analysis/ssd_analysis.py --cas 14437-17-3
python analysis/ssd_mc_uncertainty.py --cas 14437-17-3
echo ""

# --- HC20 comparison across all chemicals (Figures 10-11) ---
echo ">>> HC20 all-chemicals comparison (Figures 10-11)"
python analysis/ssd_mc_uncertainty.py --all-hcx --percentiles 20
python analysis/hcx_plots.py
echo ""

echo "============================================================"
echo "  All figures and tables generated successfully."
echo "============================================================"
echo ""
echo "Outputs:"
echo "  outputs/figures/dataset/       - Tables 1-2, Figures 1-2"
echo "  outputs/figures/analyze_results/ - Figures 3-4"
echo "  outputs/figures/uncertainty_calibration/ - Figures 5-6"
echo "  outputs/figures/ssd_analysis/  - Figures 7-11"
