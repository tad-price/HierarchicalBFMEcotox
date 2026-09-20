"""
analyze_results.py - Analysis of Hierarchical BFM Results

This script analyzes the out-of-fold predictions from the hierarchical BFM model,
generating diagnostic plots and statistics for:
- predicted vs measured correlation
- residual bias diagnostics

Requires:
- outputs/models/oof_mean.npy
- outputs/models/oof_epistemic.npy
- outputs/models/oof_aleatoric.npy

Writes to outputs/figures/.

Usage:
    python analysis/analyze_results.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Output directory for plots
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

from data.load_ecotox import load_ecotox_data


def load_data_and_predictions():
    """Load ecotox data and pre-computed OOF predictions with uncertainties."""
    print("Loading Data...")
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,  # Use log mg/L for interpretability
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )

    # Load Artifacts
    MODELS_DIR = ROOT_DIR / "outputs" / "models"
    try:
        oof_mean = np.load(MODELS_DIR / "oof_mean.npy")
        oof_epistemic = np.load(MODELS_DIR / "oof_epistemic.npy")
        oof_aleatoric = np.load(MODELS_DIR / "oof_aleatoric.npy")
        print("Loaded OOF predictions and uncertainties.")
    except FileNotFoundError:
        print(f"Artifacts not found in {MODELS_DIR}!")
        print("Please run scripts/train_bfm.py first.")
        sys.exit(1)

    df = full_data.copy()
    df["y_true"] = y_centered + y_mean
    df["y_pred"] = oof_mean + y_mean
    df["epistemic_var"] = oof_epistemic
    df["aleatoric_var"] = oof_aleatoric
    df["total_var"] = df["epistemic_var"] + df["aleatoric_var"]
    
    return df


def print_summary_statistics(df):
    """Print key summary statistics for model performance and uncertainties."""
    rmse = np.sqrt(np.mean((df["y_pred"] - df["y_true"])**2))
    mae = np.mean(np.abs(df["y_pred"] - df["y_true"]))
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"N observations: {len(df)}")
    
    print(f"\n{'='*50}")
    print("UNCERTAINTY SUMMARY (Variance)")
    print(f"{'='*50}")
    print(df[["aleatoric_var", "epistemic_var", "total_var"]].describe())
    
    print(f"\n{'='*50}")
    print("UNCERTAINTY SUMMARY (Standard Deviation)")
    print(f"{'='*50}")
    sd_summary = pd.DataFrame({
        "aleatoric_sd": np.sqrt(df["aleatoric_var"]),
        "epistemic_sd": np.sqrt(df["epistemic_var"]),
        "total_sd": np.sqrt(df["total_var"])
    })
    print(sd_summary.describe())


def plot_predicted_vs_measured_correlation(df, duration_hours=48):
    """
    Create a correlation plot of model predicted toxicities vs measured toxicities.
    
    For chemicals with measurements:
    - Filter to specified duration (default 48hr)
    - Take the mean observation and mean prediction per chemical/species
    - Scatter the results and compute correlation statistics
    
    Args:
        df: DataFrame with y_true, y_pred, CAS, species, duration columns
        duration_hours: Duration to filter to (default 48)
    """
    print(f"\n{'='*60}")
    print(f"PREDICTED vs MEASURED TOXICITY CORRELATION ({duration_hours}h)")
    print(f"{'='*60}")
    
    # Filter to specified duration
    df_filtered = df[df["duration"].astype(int) == duration_hours].copy()
    print(f"Observations at {duration_hours}h: {len(df_filtered):,}")
    
    if len(df_filtered) == 0:
        print(f"ERROR: No observations found at {duration_hours}h duration!")
        return None
    
    # Aggregate to mean per chemical/species combination
    agg_df = df_filtered.groupby(["CAS", "species"], observed=True).agg({
        "y_true": "mean",
        "y_pred": "mean"
    }).reset_index()
    
    n_pairs = len(agg_df)
    n_chemicals = agg_df["CAS"].nunique()
    n_species = agg_df["species"].nunique()
    
    print(f"Aggregated to {n_pairs:,} unique chemical/species pairs")
    print(f"   Unique chemicals: {n_chemicals:,}")
    print(f"   Unique species: {n_species:,}")
    
    # Extract arrays for plotting
    y_measured = agg_df["y_true"].values
    y_predicted = agg_df["y_pred"].values
    
    # Calculate correlation statistics
    r, p_value = pearsonr(y_measured, y_predicted)
    r_squared = r ** 2
    rmse = np.sqrt(np.mean((y_predicted - y_measured) ** 2))
    mae = np.mean(np.abs(y_predicted - y_measured))
    
    print("\nCorrelation Statistics:")
    print(f"   Pearson r:  {r:.4f}")
    print(f"   R²:         {r_squared:.4f}")
    print(f"   p-value:    {p_value:.2e}")
    print(f"   RMSE:       {rmse:.4f}")
    print(f"   MAE:        {mae:.4f}")
    
    # Create the correlation plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(y_measured, y_predicted, c='#1f77b4', s=15, alpha=0.5, 
               edgecolors='none', label=f'Chemical/Species pairs (N={n_pairs:,})')
    
    # Identity line (y = x)
    min_val = -6  # Fixed lower bound
    max_val = max(y_measured.max(), y_predicted.max())
    margin = (max_val - min_val) * 0.02
    line_range = [min_val, max_val + margin]
    ax.plot(line_range, line_range, 'k--', linewidth=1.5, alpha=0.7, label='Identity (y = x)')
    
    # Formatting
    ax.set_xlabel("Measured Toxicity (Log mg/L)", fontsize=12)
    ax.set_ylabel("Predicted Toxicity (Log mg/L)", fontsize=12)
    ax.set_title(f"Hierarchical BFM: Predicted vs Measured Toxicity ({duration_hours}h)\n"
                 f"Mean per Chemical/Species", fontsize=14)
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = (f"Pearson r = {r:.3f}\n"
                  f"N = {n_pairs:,}")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace', bbox=props)
    
    ax.legend(loc='lower right', fontsize=10)
    
    # Save figure
    output_path = OUTPUT_DIR / f"predicted_vs_measured_{duration_hours}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return {
        'r': r,
        'r_squared': r_squared,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'n_pairs': n_pairs,
        'n_chemicals': n_chemicals,
        'n_species': n_species
    }


def analyze_prediction_bias(df, duration_hours=48):
    """
    Analyze systematic bias in the model's predictions.
    
    Examines:
    1. Overall mean bias (predicted - measured)
    2. Bias as a function of toxicity level (binned analysis)
    3. Residual distribution (skewness, kurtosis)
    4. Visual diagnostics: residual plots
    
    Args:
        df: DataFrame with y_true, y_pred, CAS, species, duration columns
        duration_hours: Duration to filter to (default 48)
    """
    from scipy.stats import skew, kurtosis, ttest_1samp
    
    print(f"\n{'='*60}")
    print(f"SYSTEMATIC BIAS ANALYSIS ({duration_hours}h)")
    print(f"{'='*60}")
    
    # Filter to specified duration
    df_filtered = df[df["duration"].astype(int) == duration_hours].copy()
    
    if len(df_filtered) == 0:
        print(f"ERROR: No observations found at {duration_hours}h duration!")
        return None
    
    # Aggregate to mean per chemical/species combination
    agg_df = df_filtered.groupby(["CAS", "species"], observed=True).agg({
        "y_true": "mean",
        "y_pred": "mean"
    }).reset_index()
    
    # Calculate residuals (predicted - measured)
    # Positive = overprediction, Negative = underprediction
    y_measured = agg_df["y_true"].values
    y_predicted = agg_df["y_pred"].values
    residuals = y_predicted - y_measured
    
    n_pairs = len(agg_df)
    
    # =========================================================================
    # 1. OVERALL BIAS STATISTICS
    # =========================================================================
    print(f"\n{'='*50}")
    print("1. OVERALL BIAS STATISTICS")
    print(f"{'='*50}")
    
    mean_bias = np.mean(residuals)
    median_bias = np.median(residuals)
    std_residuals = np.std(residuals)
    
    # One-sample t-test: is mean bias significantly different from 0?
    t_stat, p_value_bias = ttest_1samp(residuals, 0)
    
    print(f"Mean Bias (Pred - Meas):     {mean_bias:+.4f}")
    print(f"Median Bias:                 {median_bias:+.4f}")
    print(f"Std Dev of Residuals:        {std_residuals:.4f}")
    print("")
    print("T-test (H0: mean bias = 0):")
    print(f"   t-statistic:              {t_stat:.4f}")
    print(f"   p-value:                  {p_value_bias:.2e}")
    
    if p_value_bias < 0.05:
        if mean_bias > 0:
            print("   → Significant POSITIVE bias (model overpredicts toxicity)")
        else:
            print("   → Significant NEGATIVE bias (model underpredicts toxicity)")
    else:
        print("   → No significant systematic bias detected")
    
    # Percentage over/under predictions
    n_over = np.sum(residuals > 0)
    n_under = np.sum(residuals < 0)
    pct_over = 100 * n_over / n_pairs
    pct_under = 100 * n_under / n_pairs
    
    print(f"\nOverpredictions:  {n_over:,} ({pct_over:.1f}%)")
    print(f"Underpredictions: {n_under:,} ({pct_under:.1f}%)")
    
    # =========================================================================
    # 2. BIAS VS TOXICITY LEVEL (binned analysis)
    # =========================================================================
    print(f"\n{'='*50}")
    print("2. BIAS BY TOXICITY LEVEL")
    print(f"{'='*50}")
    
    # Create toxicity bins
    bin_edges = [-np.inf, -4, -2, 0, 2, np.inf]
    bin_labels = ["Very Low (<-4)", "Low (-4 to -2)", "Medium (-2 to 0)", 
                  "High (0 to 2)", "Very High (>2)"]
    
    agg_df["tox_bin"] = pd.cut(agg_df["y_true"], bins=bin_edges, labels=bin_labels)
    agg_df["residual"] = residuals
    
    bin_stats = agg_df.groupby("tox_bin", observed=True).agg({
        "residual": ["mean", "std", "count"]
    }).round(4)
    bin_stats.columns = ["Mean Bias", "Std Dev", "Count"]
    
    print(f"\n{'Toxicity Level':<20} {'Mean Bias':>12} {'Std Dev':>12} {'Count':>10}")
    print("-" * 56)
    for idx, row in bin_stats.iterrows():
        bias_str = f"{row['Mean Bias']:+.4f}" if pd.notna(row['Mean Bias']) else "N/A"
        std_str = f"{row['Std Dev']:.4f}" if pd.notna(row['Std Dev']) else "N/A"
        print(f"{idx:<20} {bias_str:>12} {std_str:>12} {int(row['Count']):>10}")
    
    # =========================================================================
    # 3. RESIDUAL DISTRIBUTION ANALYSIS
    # =========================================================================
    print(f"\n{'='*50}")
    print("3. RESIDUAL DISTRIBUTION")
    print(f"{'='*50}")
    
    residual_skew = skew(residuals)
    residual_kurtosis = kurtosis(residuals)  # excess kurtosis (normal = 0)
    
    print(f"Skewness:  {residual_skew:+.4f}", end="")
    if abs(residual_skew) < 0.5:
        print(" (approximately symmetric)")
    elif residual_skew > 0:
        print(" (right-skewed: more extreme overpredictions)")
    else:
        print(" (left-skewed: more extreme underpredictions)")
    
    print(f"Kurtosis:  {residual_kurtosis:+.4f}", end="")
    if abs(residual_kurtosis) < 1:
        print(" (approximately normal tails)")
    elif residual_kurtosis > 0:
        print(" (heavy tails: more outliers than normal)")
    else:
        print(" (light tails: fewer outliers than normal)")
    
    # Percentile analysis
    print("\nResidual Percentiles:")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pct_values = np.percentile(residuals, percentiles)
    for p, v in zip(percentiles, pct_values):
        print(f"   {p:2d}th percentile: {v:+.4f}")
    
    # =========================================================================
    # 4. DIAGNOSTIC PLOTS
    # =========================================================================
    print(f"\n{'='*50}")
    print("4. GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*50}")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Plot 1: Residuals vs Predicted ---
    ax1 = axes[0]
    ax1.scatter(y_predicted, residuals, c='#1f77b4', s=10, alpha=0.4, edgecolors='none')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax1.axhline(mean_bias, color='red', linestyle='-', linewidth=1.5, 
                label=f'Mean bias = {mean_bias:+.3f}')
    ax1.set_xlabel("Predicted Toxicity (Log mg/L)", fontsize=11)
    ax1.set_ylabel("Residual (Predicted - Measured)", fontsize=11)
    ax1.set_title("Residuals vs Predicted", fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-6, None)
    
    # --- Plot 2: Residuals vs Measured ---
    ax2 = axes[1]
    ax2.scatter(y_measured, residuals, c='#2ca02c', s=10, alpha=0.4, edgecolors='none')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.axhline(mean_bias, color='red', linestyle='-', linewidth=1.5,
                label=f'Mean bias = {mean_bias:+.3f}')
    ax2.set_xlabel("Measured Toxicity (Log mg/L)", fontsize=11)
    ax2.set_ylabel("Residual (Predicted - Measured)", fontsize=11)
    ax2.set_title("Residuals vs Measured", fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, None)
    
    # --- Plot 3: Residual Histogram ---
    ax3 = axes[2]
    ax3.hist(residuals, bins=50, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero')
    ax3.axvline(mean_bias, color='red', linestyle='-', linewidth=2, 
                label=f'Mean = {mean_bias:+.3f}')
    ax3.axvline(median_bias, color='orange', linestyle='-', linewidth=2,
                label=f'Median = {median_bias:+.3f}')
    ax3.set_xlabel("Residual (Predicted - Measured)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Residual Distribution", fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-4, 4)
    
    plt.suptitle(f"Bias Diagnostic Plots ({duration_hours}h, N={n_pairs:,})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"bias_analysis_{duration_hours}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*50}")
    print("BIAS ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    print(f"Mean Bias:        {mean_bias:+.4f} log mg/L")
    print(f"   Interpretation: Model predictions are on average {abs(mean_bias):.3f} log units")
    if mean_bias > 0:
        print("                   HIGHER than measurements (overpredicts toxicity)")
    else:
        print("                   LOWER than measurements (underpredicts toxicity)")
    
    # Convert to fold-change for interpretability
    fold_change = 10 ** abs(mean_bias)
    print(f"\n   In linear scale: ~{fold_change:.2f}x difference")
    
    return {
        'mean_bias': mean_bias,
        'median_bias': median_bias,
        'std_residuals': std_residuals,
        't_stat': t_stat,
        'p_value': p_value_bias,
        'skewness': residual_skew,
        'kurtosis': residual_kurtosis,
        'pct_overprediction': pct_over,
        'bin_stats': bin_stats
    }


def main():
    df = load_data_and_predictions()
    print_summary_statistics(df)
    
    plot_predicted_vs_measured_correlation(df, duration_hours=48)
    
    analyze_prediction_bias(df, duration_hours=48)


if __name__ == "__main__":
    main()

