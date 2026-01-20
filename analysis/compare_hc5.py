"""
compare_hc5.py - Compare Traditional vs BFM Monte Carlo HC5 values

This script calculates HC5 for all chemicals at 48h using:
1. Traditional method: Lognormal fit to observed toxicity values
2. BFM MC method: Monte Carlo sampling from model predictions

Requires:
- outputs/models/full_predictions.parquet

Outputs:
- outputs/figures/hc5_comparison_{duration}h.csv
- outputs/figures/hc5_correlation_{duration}h.png

Usage:
    python analysis/compare_hc5.py
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
MIN_SPECIES_TRADITIONAL = 5  # Minimum species required for traditional HC5
N_MC_SAMPLES = 1000
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures" / "compare_hc5"
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


def calculate_traditional_hc5(df_obs_chem):
    """
    Calculate HC5 using traditional lognormal fit to observations.
    
    Returns:
        hc5: HC5 value (log mg/L), or None if insufficient data
        n_species: Number of species used
        mu: Fitted mean
        sigma: Fitted standard deviation
    """
    # Aggregate to one value per species
    species_tox = df_obs_chem.groupby("species", observed=True)["y_true"].mean()
    species_tox = species_tox.dropna()
    
    n_species = len(species_tox)
    if n_species < MIN_SPECIES_TRADITIONAL:
        return None, n_species, None, None
    
    tox_values = species_tox.values
    mu, sigma = norm.fit(tox_values)
    hc5 = norm.ppf(0.05, loc=mu, scale=sigma)
    
    return hc5, n_species, mu, sigma


def calculate_mc_hc5(pred_df_chem, n_samples=1000, seed=42):
    """
    Calculate HC5 using Monte Carlo sampling from BFM predictions.
    
    Returns:
        hc5_median: Median HC5 value
        hc5_lower: 2.5th percentile (95% CI lower)
        hc5_upper: 97.5th percentile (95% CI upper)
        n_species: Number of species
    """
    np.random.seed(seed)
    
    n_species = len(pred_df_chem)
    if n_species == 0:
        return None, None, None, 0
    
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
    
    # HC5 is at 5th percentile rank
    idx_hc5 = int(0.05 * n_species)
    hc5_samples = sorted_samples[:, idx_hc5]
    
    hc5_median = np.median(hc5_samples)
    hc5_lower = np.percentile(hc5_samples, 2.5)
    hc5_upper = np.percentile(hc5_samples, 97.5)
    
    return hc5_median, hc5_lower, hc5_upper, n_species


def main():
    print("="*60)
    print("HC5 COMPARISON: Traditional vs BFM Monte Carlo")
    print("="*60)
    
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
    
    # Calculate HC5 for each chemical
    print(f"\nCalculating HC5 for {len(common_chemicals)} chemicals...")
    
    results = []
    for cas in tqdm(common_chemicals, desc="Processing chemicals"):
        # Get data for this chemical
        df_obs_chem = df_obs_48h[df_obs_48h["CAS"] == cas]
        pred_df_chem = pred_df_48h[pred_df_48h["CAS"] == cas]
        
        # Traditional HC5
        hc5_trad, n_species_trad, mu_trad, sigma_trad = calculate_traditional_hc5(df_obs_chem)
        
        # MC HC5
        hc5_mc, hc5_mc_lower, hc5_mc_upper, n_species_mc = calculate_mc_hc5(pred_df_chem, N_MC_SAMPLES)
        
        # Get chemical name
        chem_name = get_chemical_name(df_obs, cas)
        
        results.append({
            "CAS": cas,
            "chemical_name": chem_name,
            "n_species_observed": n_species_trad,
            "n_species_predicted": n_species_mc,
            "hc5_traditional": hc5_trad,
            "hc5_trad_mu": mu_trad,
            "hc5_trad_sigma": sigma_trad,
            "hc5_mc_median": hc5_mc,
            "hc5_mc_lower_95": hc5_mc_lower,
            "hc5_mc_upper_95": hc5_mc_upper,
            "hc5_mc_ci_width": hc5_mc_upper - hc5_mc_lower if hc5_mc is not None else None
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter to chemicals with valid traditional HC5 (>= MIN_SPECIES)
    valid_df = results_df[results_df["hc5_traditional"].notna()].copy()
    print(f"\nChemicals with valid traditional HC5 (>= {MIN_SPECIES_TRADITIONAL} species): {len(valid_df)}")
    
    # Save CSV
    csv_path = OUTPUT_DIR / f"hc5_comparison_{DURATION_HOURS}h.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Create correlation plot
    print("\nCreating correlation plot...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter with error bars
    ax.errorbar(
        valid_df["hc5_traditional"], 
        valid_df["hc5_mc_median"],
        yerr=[valid_df["hc5_mc_median"] - valid_df["hc5_mc_lower_95"],
              valid_df["hc5_mc_upper_95"] - valid_df["hc5_mc_median"]],
        fmt='o', alpha=0.5, markersize=4, elinewidth=0.5, capsize=0,
        label='Chemicals with 95% CI'
    )
    
    # 1:1 line
    min_val = min(valid_df["hc5_traditional"].min(), valid_df["hc5_mc_median"].min()) - 0.5
    max_val = max(valid_df["hc5_traditional"].max(), valid_df["hc5_mc_median"].max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 Line')
    
    # Calculate correlation
    correlation = valid_df["hc5_traditional"].corr(valid_df["hc5_mc_median"])
    mean_diff = (valid_df["hc5_mc_median"] - valid_df["hc5_traditional"]).mean()
    
    # Labels
    ax.set_xlabel("Traditional HC5 (Log mg/L)", fontsize=12)
    ax.set_ylabel("BFM Monte Carlo HC5 (Log mg/L)", fontsize=12)
    ax.set_title(f"HC5 Comparison: Traditional vs BFM Monte Carlo\n"
                 f"N={len(valid_df)} chemicals at {DURATION_HOURS}h, r={correlation:.3f}, Mean Δ={mean_diff:.3f}",
                 fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    plot_path = OUTPUT_DIR / f"hc5_correlation_{DURATION_HOURS}h.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Correlation (r):           {correlation:.3f}")
    print(f"Mean difference (MC-Trad): {mean_diff:.3f}")
    print(f"Std of difference:         {(valid_df['hc5_mc_median'] - valid_df['hc5_traditional']).std():.3f}")
    
    n_mc_lower = (valid_df["hc5_mc_median"] < valid_df["hc5_traditional"]).sum()
    print(f"MC more conservative:      {n_mc_lower}/{len(valid_df)} ({100*n_mc_lower/len(valid_df):.1f}%)")
    
    print(f"\nMean 95% CI width: {valid_df['hc5_mc_ci_width'].mean():.3f}")
    print(f"Median 95% CI width: {valid_df['hc5_mc_ci_width'].median():.3f}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
