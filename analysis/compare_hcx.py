"""
compare_hcx.py - Compare Traditional vs BFM Monte Carlo HCx values

This script calculates HCx (HC5, HC20, HC50, HC80, HC95) for all chemicals at 48h using:
1. Traditional method: Normal fit to observed toxicity values (already in log scale)
2. BFM MC method: Monte Carlo sampling from model predictions

Generates:
- Individual HCx correlation plots
- Combined 4-panel figure (HC20, HC50, HC80, HC95)
- CSV with all HCx values

Usage:
    python analysis/compare_hcx.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm.auto import tqdm

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data

# Configuration
DURATION_HOURS = 48
MIN_SPECIES_TRADITIONAL = 5  # Minimum species required for traditional HCx
N_MC_SAMPLES = 1000
HCX_PERCENTILES = [5, 20, 50, 80, 95]  # HCx percentiles to calculate

OUTPUT_DIR = ROOT_DIR / "outputs" / "figures" / "compare_hcx"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """Load observations and full predictions."""
    print("Loading data...")
    
    # Load observations
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    df_obs = full_data.copy()
    df_obs["y_true"] = y_centered
    
    # Load full predictions
    MODELS_DIR = ROOT_DIR / "outputs" / "models"
    pred_path = MODELS_DIR / "full_predictions.parquet"
    
    if not pred_path.exists():
        print(f"ERROR: Full predictions not found at {pred_path}")
        print("Please run scripts/generate_predictions.py first!")
        sys.exit(1)
    
    pred_df = pd.read_parquet(pred_path)
    
    print(f"   Observations: {len(df_obs):,}")
    print(f"   Predictions: {len(pred_df):,}")
    
    return df_obs, pred_df


def get_chemical_name(df_obs, cas):
    """Get chemical name from CAS number."""
    if "chem_name" in df_obs.columns:
        chem_data = df_obs[df_obs["CAS"] == cas]
        if len(chem_data) > 0:
            name = chem_data["chem_name"].iloc[0]
            if pd.notna(name) and name.strip():
                return name.strip()
    return f"CAS {cas}"


def calculate_traditional_hcx(df_obs_chem, percentiles):
    """
    Calculate HCx using traditional normal fit to observations.
    
    Args:
        df_obs_chem: DataFrame with observations for one chemical
        percentiles: List of percentiles to calculate (e.g., [5, 20, 50, 80, 95])
    
    Returns:
        dict: {percentile: hcx_value} for each percentile, plus metadata
    """
    # Aggregate to one value per species
    species_tox = df_obs_chem.groupby("species", observed=True)["y_true"].mean()
    species_tox = species_tox.dropna()
    
    n_species = len(species_tox)
    result = {"n_species": n_species}
    
    if n_species < MIN_SPECIES_TRADITIONAL:
        for p in percentiles:
            result[f"hc{p}"] = None
        result["mu"] = None
        result["sigma"] = None
        return result
    
    tox_values = species_tox.values
    mu, sigma = norm.fit(tox_values)
    result["mu"] = mu
    result["sigma"] = sigma
    
    for p in percentiles:
        result[f"hc{p}"] = norm.ppf(p / 100, loc=mu, scale=sigma)
    
    return result


def calculate_mc_hcx(pred_df_chem, percentiles, n_samples=1000, seed=42):
    """
    Calculate HCx using Monte Carlo sampling from BFM predictions.
    
    Returns:
        dict: For each percentile, contains median, lower, upper (95% CI)
    """
    np.random.seed(seed)
    
    n_species = len(pred_df_chem)
    result = {"n_species": n_species}
    
    if n_species == 0:
        for p in percentiles:
            result[f"hc{p}_median"] = None
            result[f"hc{p}_lower"] = None
            result[f"hc{p}_upper"] = None
        return result
    
    means = pred_df_chem["pred_mean"].values
    sds = pred_df_chem["pred_total_sd"].values
    
    # Monte Carlo sampling
    samples = np.random.normal(
        loc=means[np.newaxis, :],
        scale=sds[np.newaxis, :],
        size=(n_samples, n_species)
    )
    
    # Sort each sample
    sorted_samples = np.sort(samples, axis=1)
    
    for p in percentiles:
        # HCx is at p-th percentile rank
        idx = max(0, int((p / 100) * n_species) - 1)
        hcx_samples = sorted_samples[:, idx]
        
        result[f"hc{p}_median"] = np.median(hcx_samples)
        result[f"hc{p}_lower"] = np.percentile(hcx_samples, 2.5)
        result[f"hc{p}_upper"] = np.percentile(hcx_samples, 97.5)
    
    return result


def create_combined_plot(results_df, percentiles_to_plot):
    """Create a 2x2 combined plot for multiple HCx percentiles."""
    
    n_plots = len(percentiles_to_plot)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, p in enumerate(percentiles_to_plot):
        ax = axes[idx]
        
        # Filter to valid data
        trad_col = f"hc{p}_traditional"
        mc_col = f"hc{p}_mc_median"
        mc_lower = f"hc{p}_mc_lower"
        mc_upper = f"hc{p}_mc_upper"
        
        valid = results_df[trad_col].notna() & results_df[mc_col].notna()
        plot_df = results_df[valid].copy()
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, f"No valid data for HC{p}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot with error bars
        ax.errorbar(
            plot_df[trad_col], 
            plot_df[mc_col],
            yerr=[plot_df[mc_col] - plot_df[mc_lower],
                  plot_df[mc_upper] - plot_df[mc_col]],
            fmt='o', alpha=0.4, markersize=4, elinewidth=0.5, capsize=0,
            color=colors[idx % len(colors)],
            label=f'N={len(plot_df)}'
        )
        
        # 1:1 line
        all_vals = pd.concat([plot_df[trad_col], plot_df[mc_col]])
        min_val = all_vals.min() - 0.5
        max_val = all_vals.max() + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
        
        # Calculate statistics
        correlation = plot_df[trad_col].corr(plot_df[mc_col])
        mean_diff = (plot_df[mc_col] - plot_df[trad_col]).mean()
        
        # Labels
        ax.set_xlabel(f"Traditional HC{p} (Log mg/L)", fontsize=11)
        ax.set_ylabel(f"BFM MC HC{p} (Log mg/L)", fontsize=11)
        ax.set_title(f"HC{p}: r={correlation:.3f}, Δ={mean_diff:+.2f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(percentiles_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"HCx Comparison: Traditional vs BFM Monte Carlo ({DURATION_HOURS}h)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def main():
    print("="*60)
    print("HCx COMPARISON: Traditional vs BFM Monte Carlo")
    print("="*60)
    print(f"Percentiles: {HCX_PERCENTILES}")
    
    # Load data
    df_obs, pred_df = load_data()
    
    # Filter to 48h duration
    df_obs_48h = df_obs[df_obs["duration"].astype(int) == DURATION_HOURS]
    pred_df_48h = pred_df[pred_df["duration"].astype(int) == DURATION_HOURS]
    
    # Get unique chemicals that have both observations and predictions at 48h
    obs_chemicals = set(df_obs_48h["CAS"].unique())
    pred_chemicals = set(pred_df_48h["CAS"].unique())
    common_chemicals = obs_chemicals & pred_chemicals
    
    print(f"\nChemicals at {DURATION_HOURS}h:")
    print(f"   With observations: {len(obs_chemicals)}")
    print(f"   With predictions: {len(pred_chemicals)}")
    print(f"   Common: {len(common_chemicals)}")
    
    # Calculate HCx for each chemical
    print(f"\nCalculating HCx for {len(common_chemicals)} chemicals...")
    
    results = []
    for cas in tqdm(common_chemicals, desc="Processing chemicals"):
        # Get data for this chemical
        df_obs_chem = df_obs_48h[df_obs_48h["CAS"] == cas]
        pred_df_chem = pred_df_48h[pred_df_48h["CAS"] == cas]
        
        # Traditional HCx
        trad_results = calculate_traditional_hcx(df_obs_chem, HCX_PERCENTILES)
        
        # MC HCx
        mc_results = calculate_mc_hcx(pred_df_chem, HCX_PERCENTILES, N_MC_SAMPLES)
        
        # Get chemical name
        chem_name = get_chemical_name(df_obs, cas)
        
        row = {
            "CAS": cas,
            "chemical_name": chem_name,
            "n_species_observed": trad_results["n_species"],
            "n_species_predicted": mc_results["n_species"],
            "trad_mu": trad_results["mu"],
            "trad_sigma": trad_results["sigma"],
        }
        
        # Add HCx values
        for p in HCX_PERCENTILES:
            row[f"hc{p}_traditional"] = trad_results[f"hc{p}"]
            row[f"hc{p}_mc_median"] = mc_results[f"hc{p}_median"]
            row[f"hc{p}_mc_lower"] = mc_results[f"hc{p}_lower"]
            row[f"hc{p}_mc_upper"] = mc_results[f"hc{p}_upper"]
            if mc_results[f"hc{p}_upper"] is not None and mc_results[f"hc{p}_lower"] is not None:
                row[f"hc{p}_mc_ci_width"] = mc_results[f"hc{p}_upper"] - mc_results[f"hc{p}_lower"]
            else:
                row[f"hc{p}_mc_ci_width"] = None
        
        results.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter to chemicals with valid traditional HCx
    valid_df = results_df[results_df["hc5_traditional"].notna()].copy()
    print(f"\nChemicals with valid traditional HCx (>= {MIN_SPECIES_TRADITIONAL} species): {len(valid_df)}")
    
    # Save CSV
    csv_path = OUTPUT_DIR / f"hcx_comparison_{DURATION_HOURS}h.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Create combined 4-panel plot (HC20, HC50, HC80, HC95)
    print("\nCreating combined 4-panel plot...")
    fig = create_combined_plot(results_df, [20, 50, 80, 95])
    combined_path = OUTPUT_DIR / f"hcx_combined_{DURATION_HOURS}h.png"
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()
    
    # Also create HC5 plot separately
    print("Creating HC5 plot...")
    fig_hc5 = create_combined_plot(results_df, [5])
    hc5_path = OUTPUT_DIR / f"hc5_comparison_{DURATION_HOURS}h.png"
    fig_hc5.savefig(hc5_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {hc5_path}")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for p in HCX_PERCENTILES:
        trad_col = f"hc{p}_traditional"
        mc_col = f"hc{p}_mc_median"
        
        valid = valid_df[trad_col].notna() & valid_df[mc_col].notna()
        subset = valid_df[valid]
        
        if len(subset) > 0:
            corr = subset[trad_col].corr(subset[mc_col])
            mean_diff = (subset[mc_col] - subset[trad_col]).mean()
            n_mc_lower = (subset[mc_col] < subset[trad_col]).sum()
            pct_lower = 100 * n_mc_lower / len(subset)
            ci_width_mean = subset[f"hc{p}_mc_ci_width"].mean()
            
            print(f"\nHC{p}:")
            print(f"   N chemicals:      {len(subset)}")
            print(f"   Correlation:      {corr:.3f}")
            print(f"   Mean Δ (MC-Trad): {mean_diff:+.3f}")
            print(f"   MC more conserv:  {n_mc_lower}/{len(subset)} ({pct_lower:.1f}%)")
            print(f"   Mean CI width:    {ci_width_mean:.3f}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
