"""
analyze_results.py - Comprehensive Analysis of Hierarchical BFM Results

This script analyzes the out-of-fold predictions from the hierarchical BFM model,
generating diagnostic plots and statistics for:
1. Model performance (RMSE, correlation)
2. Prediction bias analysis
3. Uncertainty decomposition (aleatoric vs epistemic)
4. Aleatoric uncertainty calibration

Requires:
- outputs/models/oof_mean.npy
- outputs/models/oof_epistemic.npy
- outputs/models/oof_aleatoric.npy

Outputs figures to:
- outputs/figures/

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
    full_data, y_centered = load_ecotox_data(
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
    df["y_true"] = y_centered
    df["y_pred"] = oof_mean
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
    
    For chemicals where we have measurements:
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
    
    print(f"\nCorrelation Statistics:")
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
    
    # Statistics box (simplified)
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
    print(f"")
    print(f"T-test (H0: mean bias = 0):")
    print(f"   t-statistic:              {t_stat:.4f}")
    print(f"   p-value:                  {p_value_bias:.2e}")
    
    if p_value_bias < 0.05:
        if mean_bias > 0:
            print(f"   → Significant POSITIVE bias (model overpredicts toxicity)")
        else:
            print(f"   → Significant NEGATIVE bias (model underpredicts toxicity)")
    else:
        print(f"   → No significant systematic bias detected")
    
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
    print(f"\nResidual Percentiles:")
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
        print(f"                   HIGHER than measurements (overpredicts toxicity)")
    else:
        print(f"                   LOWER than measurements (underpredicts toxicity)")
    
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


def explore_uncertainties(df):
    """
    Exploratory analysis of aleatoric and epistemic uncertainties.
    
    Examines:
    1. Distribution of both uncertainty types
    2. Correlation between aleatoric and epistemic
    3. Relationship with number of observations (per chemical and species)
    4. Ratio of epistemic to total uncertainty
    """
    from scipy.stats import spearmanr
    
    print(f"\n{'='*70}")
    print("UNCERTAINTY EXPLORATORY ANALYSIS")
    print(f"{'='*70}")
    
    # Convert variance to standard deviation for interpretability
    df["aleatoric_sd"] = np.sqrt(df["aleatoric_var"])
    df["epistemic_sd"] = np.sqrt(df["epistemic_var"])
    df["total_sd"] = np.sqrt(df["total_var"])
    df["epistemic_ratio"] = df["epistemic_var"] / df["total_var"]
    
    # =========================================================================
    # 1. SUMMARY STATISTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print("1. UNCERTAINTY SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    print("\n--- Standard Deviation (interpretable units) ---")
    print(f"{'Metric':<20} {'Aleatoric SD':>15} {'Epistemic SD':>15} {'Total SD':>15}")
    print("-" * 68)
    print(f"{'Mean':<20} {df['aleatoric_sd'].mean():>15.4f} {df['epistemic_sd'].mean():>15.4f} {df['total_sd'].mean():>15.4f}")
    print(f"{'Median':<20} {df['aleatoric_sd'].median():>15.4f} {df['epistemic_sd'].median():>15.4f} {df['total_sd'].median():>15.4f}")
    print(f"{'Std Dev':<20} {df['aleatoric_sd'].std():>15.4f} {df['epistemic_sd'].std():>15.4f} {df['total_sd'].std():>15.4f}")
    print(f"{'Min':<20} {df['aleatoric_sd'].min():>15.4f} {df['epistemic_sd'].min():>15.4f} {df['total_sd'].min():>15.4f}")
    print(f"{'Max':<20} {df['aleatoric_sd'].max():>15.4f} {df['epistemic_sd'].max():>15.4f} {df['total_sd'].max():>15.4f}")
    
    print("\n--- Epistemic Ratio (Epistemic / Total Variance) ---")
    print(f"Mean:   {df['epistemic_ratio'].mean():.4f}")
    print(f"Median: {df['epistemic_ratio'].median():.4f}")
    print(f"   → On average, {100*df['epistemic_ratio'].mean():.1f}% of total variance is epistemic (reducible)")
    
    # =========================================================================
    # 2. CORRELATION BETWEEN UNCERTAINTY TYPES
    # =========================================================================
    print(f"\n{'='*60}")
    print("2. CORRELATION: ALEATORIC vs EPISTEMIC")
    print(f"{'='*60}")
    
    pearson_r, pearson_p = pearsonr(df["aleatoric_sd"], df["epistemic_sd"])
    spearman_r, spearman_p = spearmanr(df["aleatoric_sd"], df["epistemic_sd"])
    
    print(f"Pearson r:  {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"Spearman ρ: {spearman_r:.4f} (p = {spearman_p:.2e})")
    
    if abs(pearson_r) < 0.3:
        print("   → Weak correlation: uncertainty types are largely independent")
    elif abs(pearson_r) < 0.6:
        print("   → Moderate correlation between uncertainty types")
    else:
        print("   → Strong correlation: uncertainty types co-vary")
    
    # =========================================================================
    # 3. UNCERTAINTY vs OBSERVATION COUNT (BY CHEMICAL)
    # =========================================================================
    print(f"\n{'='*60}")
    print("3. UNCERTAINTY vs NUMBER OF OBSERVATIONS")
    print(f"{'='*60}")
    
    # Aggregate by chemical
    chem_stats = df.groupby("CAS", observed=True).agg({
        "aleatoric_sd": "mean",
        "epistemic_sd": "mean",
        "y_true": "count"
    }).reset_index()
    chem_stats.columns = ["CAS", "mean_aleatoric_sd", "mean_epistemic_sd", "n_obs"]
    
    # Correlation: # observations vs uncertainty
    r_aleat_chem, p_aleat_chem = spearmanr(chem_stats["n_obs"], chem_stats["mean_aleatoric_sd"])
    r_epist_chem, p_epist_chem = spearmanr(chem_stats["n_obs"], chem_stats["mean_epistemic_sd"])
    
    print("\n--- By Chemical ---")
    print(f"N chemicals: {len(chem_stats)}")
    print(f"Spearman ρ (# obs vs aleatoric SD):  {r_aleat_chem:+.4f} (p = {p_aleat_chem:.2e})")
    print(f"Spearman ρ (# obs vs epistemic SD):  {r_epist_chem:+.4f} (p = {p_epist_chem:.2e})")
    
    # Aggregate by species
    species_stats = df.groupby("species", observed=True).agg({
        "aleatoric_sd": "mean",
        "epistemic_sd": "mean",
        "y_true": "count"
    }).reset_index()
    species_stats.columns = ["species", "mean_aleatoric_sd", "mean_epistemic_sd", "n_obs"]
    
    r_aleat_sp, p_aleat_sp = spearmanr(species_stats["n_obs"], species_stats["mean_aleatoric_sd"])
    r_epist_sp, p_epist_sp = spearmanr(species_stats["n_obs"], species_stats["mean_epistemic_sd"])
    
    print("\n--- By Species ---")
    print(f"N species: {len(species_stats)}")
    print(f"Spearman ρ (# obs vs aleatoric SD):  {r_aleat_sp:+.4f} (p = {p_aleat_sp:.2e})")
    print(f"Spearman ρ (# obs vs epistemic SD):  {r_epist_sp:+.4f} (p = {p_epist_sp:.2e})")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if r_epist_chem < -0.3 or r_epist_sp < -0.3:
        print("✓ Epistemic uncertainty DECREASES with more observations (as expected)")
    else:
        print("⚠ Epistemic uncertainty does not show expected negative relationship with # obs")
    
    # =========================================================================
    # 4. GENERATE DIAGNOSTIC PLOTS
    # =========================================================================
    print(f"\n{'='*60}")
    print("4. GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # --- Plot 1: Distribution of Aleatoric SD ---
    ax1 = axes[0, 0]
    # Filter outliers for visualization
    aleat_99 = df["aleatoric_sd"].quantile(0.99)
    ax1.hist(df["aleatoric_sd"][df["aleatoric_sd"] < aleat_99], bins=50, 
             color='#e41a1c', alpha=0.7, edgecolor='white')
    ax1.axvline(df["aleatoric_sd"].median(), color='black', linestyle='--', 
                linewidth=2, label=f'Median = {df["aleatoric_sd"].median():.3f}')
    ax1.set_xlabel("Aleatoric SD", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of Aleatoric Uncertainty", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 2: Distribution of Epistemic SD ---
    ax2 = axes[0, 1]
    epist_99 = df["epistemic_sd"].quantile(0.99)
    ax2.hist(df["epistemic_sd"][df["epistemic_sd"] < epist_99], bins=50, 
             color='#377eb8', alpha=0.7, edgecolor='white')
    ax2.axvline(df["epistemic_sd"].median(), color='black', linestyle='--', 
                linewidth=2, label=f'Median = {df["epistemic_sd"].median():.3f}')
    ax2.set_xlabel("Epistemic SD", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Distribution of Epistemic Uncertainty", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 3: Aleatoric vs Epistemic Scatter ---
    ax3 = axes[0, 2]
    # Subsample for visualization if too many points
    n_plot = min(5000, len(df))
    idx = np.random.choice(len(df), n_plot, replace=False)
    ax3.scatter(df["aleatoric_sd"].iloc[idx], df["epistemic_sd"].iloc[idx], 
                c='#984ea3', s=5, alpha=0.3, edgecolors='none')
    ax3.set_xlabel("Aleatoric SD", fontsize=11)
    ax3.set_ylabel("Epistemic SD", fontsize=11)
    ax3.set_title(f"Aleatoric vs Epistemic\n(ρ = {spearman_r:.3f})", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Epistemic SD vs # Observations (by Chemical) ---
    ax4 = axes[1, 0]
    ax4.scatter(chem_stats["n_obs"], chem_stats["mean_epistemic_sd"], 
                c='#377eb8', s=20, alpha=0.5, edgecolors='none')
    ax4.set_xlabel("# Observations per Chemical", fontsize=11)
    ax4.set_ylabel("Mean Epistemic SD", fontsize=11)
    ax4.set_title(f"Epistemic Uncertainty vs Obs Count (Chemical)\nρ = {r_epist_chem:.3f}", fontsize=12)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Epistemic SD vs # Observations (by Species) ---
    ax5 = axes[1, 1]
    ax5.scatter(species_stats["n_obs"], species_stats["mean_epistemic_sd"], 
                c='#4daf4a', s=20, alpha=0.5, edgecolors='none')
    ax5.set_xlabel("# Observations per Species", fontsize=11)
    ax5.set_ylabel("Mean Epistemic SD", fontsize=11)
    ax5.set_title(f"Epistemic Uncertainty vs Obs Count (Species)\nρ = {r_epist_sp:.3f}", fontsize=12)
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # --- Plot 6: Epistemic Ratio Distribution ---
    ax6 = axes[1, 2]
    ax6.hist(df["epistemic_ratio"], bins=50, color='#ff7f00', alpha=0.7, edgecolor='white')
    ax6.axvline(df["epistemic_ratio"].median(), color='black', linestyle='--', 
                linewidth=2, label=f'Median = {df["epistemic_ratio"].median():.3f}')
    ax6.set_xlabel("Epistemic / Total Variance", fontsize=11)
    ax6.set_ylabel("Frequency", fontsize=11)
    ax6.set_title("Epistemic Ratio Distribution", fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xlim(0, 1)
    
    plt.suptitle("Uncertainty Exploratory Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "uncertainty_exploration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"• Aleatoric SD: median = {df['aleatoric_sd'].median():.3f}, mean = {df['aleatoric_sd'].mean():.3f}")
    print(f"• Epistemic SD: median = {df['epistemic_sd'].median():.3f}, mean = {df['epistemic_sd'].mean():.3f}")
    print(f"• Epistemic accounts for {100*df['epistemic_ratio'].median():.1f}% of total variance (median)")
    print(f"• Aleatoric-Epistemic correlation: ρ = {spearman_r:.3f}")
    print(f"• Epistemic-Observations correlation: ρ = {r_epist_chem:.3f} (chemicals), ρ = {r_epist_sp:.3f} (species)")
    
    return {
        'aleatoric_sd_median': df['aleatoric_sd'].median(),
        'epistemic_sd_median': df['epistemic_sd'].median(),
        'epistemic_ratio_median': df['epistemic_ratio'].median(),
        'aleat_epist_correlation': spearman_r,
        'epist_obs_corr_chemical': r_epist_chem,
        'epist_obs_corr_species': r_epist_sp
    }


def check_aleatoric_calibration(df, min_replicates=5):
    """
    Check calibration of predicted aleatoric uncertainty against empirical observation variability.
    
    Methodology:
    - For each (chemical, species, duration) group with ≥ min_replicates observations,
      the empirical SD of y_true represents actual measurement variability.
    - The model's predicted aleatoric SD should match this empirical SD if well-calibrated.
    - We compare predicted vs empirical SD and compute calibration metrics.
    
    Mathematically:
    - Aleatoric uncertainty represents irreducible noise in the data
    - For a well-calibrated model: predicted_aleatoric_sd ≈ empirical_sd(y_true)
    - Calibration ratio = predicted_sd / empirical_sd (should be ≈ 1.0)
    
    Args:
        df: DataFrame with y_true, aleatoric_var, CAS, species, duration columns
        min_replicates: Minimum number of observations per group for reliable SD estimate
    """
    print(f"\n{'='*70}")
    print("ALEATORIC UNCERTAINTY CALIBRATION CHECK")
    print(f"{'='*70}")
    print(f"\nMethodology:")
    print(f"  • For groups with ≥{min_replicates} replicate measurements,")
    print(f"    compare predicted aleatoric SD to empirical SD of observations")
    print(f"  • If well-calibrated: predicted SD ≈ empirical SD (ratio ≈ 1.0)")
    
    # Add aleatoric SD column
    df["aleatoric_sd"] = np.sqrt(df["aleatoric_var"])
    
    # Group by chemical, species, duration
    grouped = df.groupby(["CAS", "species", "duration"], observed=True).agg({
        "y_true": ["std", "count", "mean"],
        "aleatoric_sd": "mean"  # Should be constant within chemical, but take mean
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ["CAS", "species", "duration", "empirical_sd", "n_obs", "mean_y", "predicted_sd"]
    
    # Filter to groups with enough replicates for reliable SD
    # Note: std() returns NaN for n=1, we need n >= min_replicates
    calibration_df = grouped[
        (grouped["n_obs"] >= min_replicates) & 
        (grouped["empirical_sd"].notna()) &
        (grouped["empirical_sd"] > 0)  # Avoid division by zero
    ].copy()
    
    n_groups = len(calibration_df)
    n_total = len(grouped)
    
    print(f"\n{'='*60}")
    print("1. DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total (chemical, species, duration) groups: {n_total:,}")
    print(f"Groups with ≥{min_replicates} replicates: {n_groups:,} ({100*n_groups/n_total:.1f}%)")
    print(f"Median replicates in analysis: {calibration_df['n_obs'].median():.0f}")
    print(f"Max replicates: {calibration_df['n_obs'].max():.0f}")
    
    if n_groups < 10:
        print(f"\n⚠ Warning: Too few groups ({n_groups}) for reliable calibration analysis!")
        return None
    
    # Calculate calibration ratio
    calibration_df["ratio"] = calibration_df["predicted_sd"] / calibration_df["empirical_sd"]
    
    # =========================================================================
    # 2. CALIBRATION STATISTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print("2. CALIBRATION STATISTICS")
    print(f"{'='*60}")
    
    # Correlation between predicted and empirical SD
    r_pearson, p_pearson = pearsonr(calibration_df["predicted_sd"], calibration_df["empirical_sd"])
    
    print(f"\nCorrelation (predicted vs empirical SD):")
    print(f"   Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    
    # Mean and median calibration ratio
    mean_ratio = calibration_df["ratio"].mean()
    median_ratio = calibration_df["ratio"].median()
    
    print(f"\nCalibration Ratio (Predicted / Empirical):")
    print(f"   Mean ratio:   {mean_ratio:.4f}")
    print(f"   Median ratio: {median_ratio:.4f}")
    
    # Interpret
    if 0.8 <= median_ratio <= 1.2:
        print(f"   ✓ Well calibrated (ratio near 1.0)")
    elif median_ratio < 0.8:
        print(f"   ⚠ Under-confident: predicted SD < empirical SD")
    else:
        print(f"   ⚠ Over-confident: predicted SD > empirical SD")
    
    # What fraction are within reasonable bounds?
    within_50pct = ((calibration_df["ratio"] >= 0.5) & (calibration_df["ratio"] <= 2.0)).mean()
    
    print(f"\nCalibration Quality:")
    print(f"   {100*within_50pct:.1f}% of groups have ratio within [0.5, 2.0]")
    
    # Distribution of ratios
    print(f"\nRatio Percentiles:")
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = np.percentile(calibration_df["ratio"], p)
        print(f"   {p:2d}th percentile: {val:.3f}")
    
    # =========================================================================
    # 3. CALIBRATION BY MEASUREMENT COUNT
    # =========================================================================
    print(f"\n{'='*60}")
    print("3. CALIBRATION BY REPLICATE COUNT")
    print(f"{'='*60}")
    
    # More replicates = more reliable empirical SD
    # Check if calibration improves with more replicates
    bins = [5, 10, 20, 50, 100, float('inf')]
    labels = ["5-9", "10-19", "20-49", "50-99", "100+"]
    
    calibration_df["n_bin"] = pd.cut(calibration_df["n_obs"], bins=bins, labels=labels, right=False)
    
    print(f"\n{'Replicates':<12} {'N groups':>10} {'Mean Ratio':>12} {'Median Ratio':>14} {'Mean Pred SD':>14} {'Mean Emp SD':>12}")
    print("-" * 78)
    
    for label in labels:
        subset = calibration_df[calibration_df["n_bin"] == label]
        if len(subset) > 0:
            print(f"{label:<12} {len(subset):>10} {subset['ratio'].mean():>12.3f} {subset['ratio'].median():>14.3f} {subset['predicted_sd'].mean():>14.3f} {subset['empirical_sd'].mean():>12.3f}")
    
    # =========================================================================
    # 4. DIAGNOSTIC PLOTS
    # =========================================================================
    print(f"\n{'='*60}")
    print("4. GENERATING CALIBRATION PLOTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Plot 1: Predicted vs Empirical SD ---
    ax1 = axes[0]
    
    # Color by number of replicates (more = more reliable)
    sc = ax1.scatter(calibration_df["empirical_sd"], calibration_df["predicted_sd"], 
                     c=np.log10(calibration_df["n_obs"]), cmap='viridis', 
                     s=15, alpha=0.6, edgecolors='none')
    
    # Identity line
    max_val = max(calibration_df["empirical_sd"].max(), calibration_df["predicted_sd"].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Perfect calibration (y=x)')
    
    ax1.set_xlabel("Empirical SD (from replicates)", fontsize=11)
    ax1.set_ylabel("Predicted Aleatoric SD", fontsize=11)
    ax1.set_title(f"Aleatoric Calibration\n(r = {r_pearson:.3f})", fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    
    plt.colorbar(sc, ax=ax1, label='log₁₀(N replicates)')
    
    # --- Plot 2: Calibration Ratio Histogram ---
    ax2 = axes[1]
    
    # Clip for visualization
    ratio_clipped = calibration_df["ratio"].clip(0, 5)
    ax2.hist(ratio_clipped, bins=50, color='#4daf4a', alpha=0.7, edgecolor='white')
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Perfect (1.0)')
    ax2.axvline(median_ratio, color='red', linestyle='-', linewidth=2, 
                label=f'Median = {median_ratio:.2f}')
    ax2.set_xlabel("Calibration Ratio (Predicted / Empirical)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Distribution of Calibration Ratios", fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(0, 5)
    
    # --- Plot 3: Ratio vs Number of Replicates ---
    ax3 = axes[2]
    
    ax3.scatter(calibration_df["n_obs"], calibration_df["ratio"], 
                c='#984ea3', s=15, alpha=0.5, edgecolors='none')
    ax3.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel("Number of Replicates", fontsize=11)
    ax3.set_ylabel("Calibration Ratio", fontsize=11)
    ax3.set_title("Calibration vs Sample Size", fontsize=12)
    ax3.set_xscale('log')
    ax3.set_ylim(0, 5)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"Aleatoric Uncertainty Calibration (N = {n_groups:,} groups with ≥{min_replicates} replicates)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "aleatoric_calibration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"• Correlation (predicted vs empirical): r = {r_pearson:.3f}")
    print(f"• Median calibration ratio: {median_ratio:.3f}")
    print(f"• {100*within_50pct:.1f}% of predictions within 2x of empirical SD")
    
    if median_ratio < 1.0:
        print(f"\n→ Model's aleatoric SD is {100*(1-median_ratio):.0f}% LOWER than empirical variability")
        print(f"  This means the model is UNDER-estimating measurement noise")
    else:
        print(f"\n→ Model's aleatoric SD is {100*(median_ratio-1):.0f}% HIGHER than empirical variability")
        print(f"  This means the model is OVER-estimating measurement noise")
    
    return {
        'n_groups': n_groups,
        'correlation': r_pearson,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'within_2x': within_50pct
    }


def main():
    df = load_data_and_predictions()
    print_summary_statistics(df)
    
    # Generate correlation plot for 48hr duration
    plot_predicted_vs_measured_correlation(df, duration_hours=48)
    
    # Analyze systematic bias
    analyze_prediction_bias(df, duration_hours=48)
    
    # Exploratory analysis of uncertainties
    explore_uncertainties(df)
    
    # Calibration check for aleatoric uncertainty
    check_aleatoric_calibration(df, min_replicates=5)


if __name__ == "__main__":
    main()
