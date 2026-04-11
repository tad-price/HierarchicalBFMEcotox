"""
ssd_analysis.py - Species Sensitivity Distribution Analysis

This script generates:
1. Traditional SSD: Lognormal fit to observed toxicity values
2. Novel SSD: Scatter model predictions for all species (target chemical/duration)
3. Novel SSD with uncertainty bars: Decomposed epistemic/aleatoric error bars
4. Combined comparison: Traditional vs novel SSD on same axes

For MCMC posterior uncertainty analysis, see ssd_mc_uncertainty.py.

Requires:
- outputs/models/full_predictions.parquet

Outputs figures to:
- outputs/figures/ssd_analysis/

Usage:
    python analysis/ssd_analysis.py --cas 1912-24-9
    python analysis/ssd_analysis.py --cas 14437-17-3
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data


# =============================================================================
# CONFIGURATION
# =============================================================================
DURATION_HOURS = 48
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# These are set by parse_args() or set_target() before any analysis runs.
TARGET_CAS = None
CHEMICAL_NAME = None


def set_target(cas, name=None):
    """Set the target chemical for this module (used by ssd_mc_uncertainty.py too)."""
    global TARGET_CAS, CHEMICAL_NAME
    TARGET_CAS = cas
    CHEMICAL_NAME = name


def load_observations():
    """Load actual observations (for traditional SSD)."""
    print("Loading observed data...")
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,  # Use log mg/L for interpretability
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )

    df = full_data.copy()
    df["y_true"] = y_centered + y_mean
    
    print(f"   Loaded {len(df):,} observations")
    return df


def load_full_predictions():
    """Load full predictions table (for novel and uncertainty SSDs)."""
    print("Loading full predictions...")
    MODELS_DIR = ROOT_DIR / "outputs" / "models"
    pred_path = MODELS_DIR / "full_predictions.parquet"
    
    if not pred_path.exists():
        print(f"ERROR: Full predictions not found at {pred_path}")
        print("Please run scripts/generate_predictions.py first!")
        sys.exit(1)
    
    pred_df = pd.read_parquet(pred_path)
    print(f"   Loaded {len(pred_df):,} predictions")
    print(f"   Unique chemicals: {pred_df['CAS'].nunique()}")
    print(f"   Unique species: {pred_df['species'].nunique()}")
    
    return pred_df


def get_chemical_name(df, cas_number):
    """Look up the chemical name from the dataframe based on CAS number."""
    if "chem_name" in df.columns:
        chem_data = df[df["CAS"] == cas_number]
        if len(chem_data) > 0:
            name = chem_data["chem_name"].iloc[0]
            if pd.notna(name) and name.strip():
                return name.strip()
    return f"CAS {cas_number}"


def filter_observations(df_obs):
    """Filter observations to target chemical at specified duration."""
    global CHEMICAL_NAME

    if CHEMICAL_NAME is None:
        CHEMICAL_NAME = get_chemical_name(df_obs, TARGET_CAS)

    df_filtered = df_obs[(df_obs["CAS"] == TARGET_CAS) &
                         (df_obs["duration"].astype(int) == DURATION_HOURS)].copy()

    n_obs = len(df_filtered)
    n_species = df_filtered["species"].nunique()
    print(f"\nFiltered observations to {CHEMICAL_NAME} (CAS {TARGET_CAS}) at {DURATION_HOURS}h:")
    print(f"   Observations: {n_obs}")
    print(f"   Unique species: {n_species}")

    return df_filtered


def filter_predictions(pred_df):
    """Filter predictions to target chemical at specified duration."""
    df_filtered = pred_df[(pred_df["CAS"] == TARGET_CAS) & 
                          (pred_df["duration"].astype(int) == DURATION_HOURS)].copy()
    
    n_species = len(df_filtered)
    print(f"\nFiltered predictions to {CHEMICAL_NAME} at {DURATION_HOURS}h:")
    print(f"   Predictions for {n_species} species")
    
    return df_filtered


# =============================================================================
# TRADITIONAL SSD (based on actual observations)
# =============================================================================

def plot_traditional_ssd(df_obs):
    """
    Generate a traditional Species Sensitivity Distribution.
    
    Methodology:
    - Aggregate to one toxicity value per species (mean of observations)
    - Since y_true is already log-transformed, fit a normal distribution
    - Plot empirical CDF (observations) and fitted normal CDF
    """
    print("\n" + "="*60)
    print("TRADITIONAL SSD (from observations)")
    print("="*60)
    
    # Aggregate to one value per species (mean of tests)
    species_tox = df_obs.groupby("species", observed=True)["y_true"].mean().reset_index()
    species_tox.columns = ["species", "toxicity"]
    species_tox = species_tox.dropna(subset=["toxicity"])
    
    n_species = len(species_tox)
    print(f"Aggregated to {n_species} species (mean toxicity per species)")
    
    # Fit normal distribution (data is already log-transformed)
    tox_values = species_tox["toxicity"].values
    mu, std = norm.fit(tox_values)
    print(f"Fitted Normal: μ = {mu:.4f}, σ = {std:.4f}")
    
    # Calculate empirical CDF plotting positions
    sorted_tox = np.sort(tox_values)
    empirical_cdf = np.arange(1, n_species + 1) / (n_species + 1)
    
    # Generate fitted curve
    x_range = np.linspace(sorted_tox.min() - 1, sorted_tox.max() + 1, 200)
    fitted_cdf = norm.cdf(x_range, loc=mu, scale=std)
    
    # Calculate HC5
    hc5_traditional = norm.ppf(0.05, loc=mu, scale=std)
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    plt.plot(x_range, fitted_cdf, 'b-', linewidth=2, 
             label=f'Fitted Normal (μ={mu:.2f}, σ={std:.2f})')
    plt.scatter(sorted_tox, empirical_cdf, c='black', s=60, zorder=5,
                label=f'Observed Species (N={n_species})')
    plt.axvline(hc5_traditional, color='red', linestyle='--', linewidth=1.5,
                label=f'HC5 = {hc5_traditional:.2f}')
    
    plt.xlabel("Toxicity (Log mg/L)", fontsize=12)
    plt.ylabel("Fraction of Species Affected", fontsize=12)
    plt.title(f"Traditional SSD: {CHEMICAL_NAME} at {DURATION_HOURS}h", fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    safe_name = CHEMICAL_NAME.lower().replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    output_path = OUTPUT_DIR / f"ssd_traditional_{safe_name}_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return hc5_traditional


# =============================================================================
# NOVEL SSD (model predictions for all species)
# =============================================================================

def plot_novel_ssd(pred_df):
    """
    Generate a novel 'predicted' Species Sensitivity Distribution.
    
    Methodology:
    - Use model predictions for the target chemical at target duration
    - This gives predictions for ALL species in the dataset
    - Sort by predicted toxicity and plot as empirical CDF
    """
    print("\n" + "="*60)
    print("NOVEL SSD (model predictions for all species)")
    print("="*60)
    
    n_species = len(pred_df)
    print(f"Using {n_species} species predictions")
    
    # Sort by predicted toxicity
    sorted_pred = np.sort(pred_df["pred_mean"].values)
    empirical_cdf = np.arange(1, n_species + 1) / (n_species + 1)
    
    # Plot
    plt.figure(figsize=(10, 7))
    
    plt.scatter(sorted_pred, empirical_cdf, c='blue', s=10, alpha=0.7,
                label=f'Predicted Species Sensitivity (N={n_species})')
    
    plt.xlabel("Predicted Toxicity (Log mg/L)", fontsize=12)
    plt.ylabel("Fraction of Species Affected", fontsize=12)
    plt.title(f"Novel SSD (Model Predictions): {CHEMICAL_NAME} at {DURATION_HOURS}h", fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    safe_name = CHEMICAL_NAME.lower().replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    output_path = OUTPUT_DIR / f"ssd_novel_{safe_name}_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_novel_ssd_with_uncertainty(pred_df, z_score=1.96):
    """
    Generate a novel SSD scatterplot with decomposed uncertainty bars.
    
    Methodology:
    - For each species, plot the predicted toxicity as a point
    - Add horizontal error bars showing:
      1. Inner bar (darker): Aleatoric uncertainty (irreducible noise, constant per chemical)
      2. Outer bar (lighter): Epistemic uncertainty (model uncertainty, varies per species)
    - Uses ±z_score standard deviations (default 1.96 for 95% CI)
    
    The aleatoric uncertainty represents measurement noise that can't be reduced.
    The epistemic uncertainty represents model uncertainty about the true toxicity,
    which could potentially be reduced with more data.
    
    Args:
        pred_df: Predictions for target chemical (one row per species)
        z_score: Number of standard deviations for error bars (default 1.96 for 95% CI)
    """
    print("\n" + "="*60)
    print("NOVEL SSD WITH UNCERTAINTY BARS")
    print("="*60)
    
    n_species = len(pred_df)
    print(f"Using {n_species} species with uncertainty estimates")
    
    # Sort by predicted mean toxicity and get sorted indices
    sorted_indices = np.argsort(pred_df["pred_mean"].values)
    pred_df_sorted = pred_df.iloc[sorted_indices].copy()
    
    # Extract uncertainty components (as standard deviations)
    means = pred_df_sorted["pred_mean"].values
    aleatoric_sd = np.sqrt(pred_df_sorted["pred_aleatoric_var"].values)  # Constant per chemical
    epistemic_sd = np.sqrt(pred_df_sorted["pred_epistemic_var"].values)  # Varies per species
    total_sd = np.sqrt(pred_df_sorted["pred_aleatoric_var"].values + 
                       pred_df_sorted["pred_epistemic_var"].values)
    
    print(f"Mean prediction range: [{means.min():.2f}, {means.max():.2f}]")
    print(f"Aleatoric SD (constant): {aleatoric_sd[0]:.3f}")
    print(f"Epistemic SD range: [{epistemic_sd.min():.3f}, {epistemic_sd.max():.3f}]")
    print(f"Total SD range: [{total_sd.min():.3f}, {total_sd.max():.3f}]")
    
    # Empirical CDF y-values
    y_cdf = np.arange(1, n_species + 1) / (n_species + 1)
    
    # Calculate error bar half-widths (±z_score * SD)
    aleatoric_err = z_score * aleatoric_sd
    total_err = z_score * total_sd  # Total includes both components
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot total uncertainty bars first (epistemic + aleatoric) - lighter color, outer bar
    # Format: xerr can be (2, N) array with [lower_errors, upper_errors] or (N,) for symmetric
    ax.errorbar(means, y_cdf, xerr=total_err, fmt='none', 
                ecolor='#6baed6', elinewidth=2.5, alpha=0.6, capsize=0,
                label=f'Epistemic + Aleatoric (±{z_score:.2f}σ)')
    
    # Plot aleatoric uncertainty bars on top - darker color, inner bar
    ax.errorbar(means, y_cdf, xerr=aleatoric_err, fmt='none',
                ecolor='#08519c', elinewidth=4, alpha=0.8, capsize=0,
                label=f'Aleatoric only (±{z_score:.2f}σ)')
    
    # Plot points on top
    ax.scatter(means, y_cdf, c='#08519c', s=15, zorder=5, edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel("Predicted Toxicity (Log mg/L)", fontsize=12)
    ax.set_ylabel("Fraction of Species Affected", fontsize=12)
    ax.set_title(f"Novel SSD with Uncertainty: {CHEMICAL_NAME} at {DURATION_HOURS}h\n"
                 f"(N={n_species} species, 95% CI shown)", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add annotation about uncertainty interpretation
    textstr = f"Aleatoric SD: {aleatoric_sd[0]:.3f} (constant)\nEpistemic SD: {epistemic_sd.mean():.3f} (mean)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    safe_name = CHEMICAL_NAME.lower().replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    output_path = OUTPUT_DIR / f"{safe_name}_novel_uncertainty.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()




def plot_traditional_vs_novel_ssd(df_obs, pred_df):
    """
    Plot traditional SSD curve and novel SSD predictions on the same panel,
    along with the raw observations from the dataset.
    
    This creates a comprehensive comparison showing:
    1. Traditional SSD: Fitted normal curve to aggregated observations
    2. Novel SSD: Model predictions for all species (empirical CDF)
    3. Observations: Raw observed toxicity values
    """
    print("\n" + "="*60)
    print("TRADITIONAL vs NOVEL SSD (same panel)")
    print("="*60)
    
    # === Traditional SSD: Fit normal to aggregated observations ===
    species_tox_obs = df_obs.groupby("species", observed=True)["y_true"].mean().reset_index()
    species_tox_obs.columns = ["species", "toxicity"]
    species_tox_obs = species_tox_obs.dropna(subset=["toxicity"])
    
    n_species_obs = len(species_tox_obs)
    tox_values_obs = species_tox_obs["toxicity"].values
    
    # Fit normal distribution
    mu_trad, std_trad = norm.fit(tox_values_obs)
    print(f"Traditional SSD: {n_species_obs} observed species")
    print(f"   Fitted Normal: μ = {mu_trad:.4f}, σ = {std_trad:.4f}")
    
    # Traditional HC5
    hc5_traditional = norm.ppf(0.05, loc=mu_trad, scale=std_trad)
    
    # === Novel SSD: Model predictions ===
    n_species_pred = len(pred_df)
    sorted_pred = np.sort(pred_df["pred_mean"].values)
    y_cdf_novel = np.arange(1, n_species_pred + 1) / (n_species_pred + 1)
    print(f"Novel SSD: {n_species_pred} predicted species")
    
    # Novel HC5 (empirical, at 5th percentile rank)
    idx_hc5 = int(0.05 * n_species_pred)
    hc5_novel = sorted_pred[idx_hc5]
    
    # === Raw observations (for scatter) ===
    sorted_obs = np.sort(tox_values_obs)
    y_cdf_obs = np.arange(1, n_species_obs + 1) / (n_species_obs + 1)
    
    # === Create plot ===
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine x-axis range for fitted curve
    x_min = min(sorted_pred.min(), sorted_obs.min()) - 1
    x_max = max(sorted_pred.max(), sorted_obs.max()) + 1
    x_range = np.linspace(x_min, x_max, 300)
    
    # Plot traditional SSD curve (fitted normal)
    fitted_cdf = norm.cdf(x_range, loc=mu_trad, scale=std_trad)
    ax.plot(x_range, fitted_cdf, 'r-', linewidth=2.5, 
            label=f'Traditional SSD (μ={mu_trad:.2f}, σ={std_trad:.2f})')
    
    # Plot novel SSD (model predictions as empirical CDF)
    ax.scatter(sorted_pred, y_cdf_novel, c='#1f77b4', s=12, alpha=0.6,
               label=f'Novel SSD - Model Predictions (N={n_species_pred})')
    
    # Plot raw observations
    ax.scatter(sorted_obs, y_cdf_obs, c='black', s=80, marker='o', 
               edgecolors='white', linewidths=1.5, zorder=5,
               label=f'Observations (N={n_species_obs})')
    
    # HC5 lines
    ax.axvline(hc5_traditional, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Traditional HC5 = {hc5_traditional:.2f}')
    ax.axvline(hc5_novel, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Novel HC5 = {hc5_novel:.2f}')
    
    ax.set_xlabel("Toxicity (Log mg/L)", fontsize=12)
    ax.set_ylabel("Fraction of Species Affected", fontsize=12)
    ax.set_title(f"Traditional vs Novel SSD: {CHEMICAL_NAME} at {DURATION_HOURS}h", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add annotation box with HC5 comparison
    textstr = (f"HC5 Comparison:\n"
               f"  Traditional: {hc5_traditional:.3f}\n"
               f"  Novel: {hc5_novel:.3f}\n"
               f"  Δ: {hc5_novel - hc5_traditional:.3f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.35, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    safe_name = CHEMICAL_NAME.lower().replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    output_path = OUTPUT_DIR / f"SSD_comparison_{safe_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return hc5_traditional, hc5_novel


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Species Sensitivity Distribution Analysis")
    parser.add_argument("--cas", type=str, default="1912-24-9",
                        help="CAS number of target chemical (default: 1912-24-9 = Atrazine)")
    parser.add_argument("--name", type=str, default=None,
                        help="Chemical name (auto-detected from data if not provided)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_target(args.cas, args.name)

    print("="*60)
    print(f"SPECIES SENSITIVITY DISTRIBUTION ANALYSIS")
    print(f"Target: CAS {TARGET_CAS} at {DURATION_HOURS}h")
    print("="*60)

    # Load data
    df_obs = load_observations()
    pred_df = load_full_predictions()

    # Filter to target chemical
    df_obs_filtered = filter_observations(df_obs)
    pred_df_filtered = filter_predictions(pred_df)

    if len(df_obs_filtered) == 0:
        print("ERROR: No observations for the specified chemical and duration!")
        sys.exit(1)

    if len(pred_df_filtered) == 0:
        print("ERROR: No predictions for the specified chemical and duration!")
        sys.exit(1)

    # Generate plots
    hc5_traditional = plot_traditional_ssd(df_obs_filtered)
    plot_novel_ssd(pred_df_filtered)
    plot_novel_ssd_with_uncertainty(pred_df_filtered)
    hc5_traditional_2, hc5_novel = plot_traditional_vs_novel_ssd(df_obs_filtered, pred_df_filtered)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Chemical: {CHEMICAL_NAME}")
    print(f"Duration: {DURATION_HOURS}h")
    print(f"Observed species: {df_obs_filtered['species'].nunique()}")
    print(f"Predicted species: {len(pred_df_filtered)}")
    print(f"\nHC5 Comparison:")
    print(f"  Traditional: {hc5_traditional:.3f} ({np.exp(hc5_traditional):.6f} mg/L)")
    print(f"  Novel:       {hc5_novel:.3f} ({np.exp(hc5_novel):.6f} mg/L)")
    print(f"\nFor MCMC uncertainty analysis, run: python analysis/ssd_mc_uncertainty.py")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
