"""
uncertainty_figures.py - Generate Uncertainty Calibration Figures for Paper

This script generates two key figures for the Uncertainty Calibration section:

1. Uncertainty vs Observation Count: Shows how epistemic and aleatoric uncertainty
   relate to the amount of data per chemical. Epistemic should decrease with more
   data (model confidence increases); aleatoric is the learned irreducible noise.

2. Example Predictions with Uncertainty: For 5 selected chemicals, shows model
   predictions with uncertainty bars compared to actual observed measurements.

Usage:
    python analysis/uncertainty_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data

# Configuration
DURATION_HOURS = 48
MIN_OBS_FOR_EXAMPLES = 10  # Minimum observations for example chemicals
N_EXAMPLE_CHEMICALS = 5
RANDOM_SEED = 42

OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_oof_data():
    """Load ecotox data and pre-computed OOF predictions with uncertainties."""
    print("Loading OOF Data...")
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )

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
    
    # Convert to SD for interpretability
    df["epistemic_sd"] = np.sqrt(df["epistemic_var"])
    df["aleatoric_sd"] = np.sqrt(df["aleatoric_var"])
    df["total_sd"] = np.sqrt(df["total_var"])
    
    return df


def load_full_predictions():
    """
    Load full predictions from the trained model.
    
    This contains predictions for ALL (chemical, species, duration) triplets,
    not just observed ones. The uncertainty estimates here represent the model's
    learned uncertainty after training on the full dataset.
    """
    print("Loading Full Predictions...")
    MODELS_DIR = ROOT_DIR / "outputs" / "models"
    pred_path = MODELS_DIR / "full_predictions.parquet"
    
    if not pred_path.exists():
        print(f"Full predictions not found at {pred_path}")
        print("Please run scripts/generate_predictions.py first!")
        sys.exit(1)
    
    pred_df = pd.read_parquet(pred_path)
    print(f"Loaded {len(pred_df):,} predictions")
    
    # Add SD columns
    pred_df["epistemic_sd"] = np.sqrt(pred_df["pred_epistemic_var"])
    pred_df["aleatoric_sd"] = np.sqrt(pred_df["pred_aleatoric_var"])
    
    return pred_df


def load_observation_counts():
    """Load observation counts per chemical from the raw data."""
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, _, _ = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    
    # Count observations per CAS at 48h
    df_48h = full_data[full_data["duration"].astype(int) == DURATION_HOURS]
    obs_counts = df_48h.groupby("CAS", observed=True).agg({
        "chem_name": "first"
    }).reset_index()
    obs_counts["n_obs"] = df_48h.groupby("CAS", observed=True).size().values
    
    return obs_counts


def plot_uncertainty_vs_observations(df):
    """
    Plot how epistemic and aleatoric uncertainty relate to observation count per chemical.
    
    Uses OUT-OF-FOLD predictions, which provide more honest uncertainty estimates
    since each observation's uncertainty was estimated by a model that did not train on it.
    
    SCIENTIFIC INTERPRETATION:
    
    Panel A: Epistemic Uncertainty
    - Epistemic uncertainty = variance across posterior samples of predictions
    - Each prediction depends on BOTH the chemical's and species' latent factors
    - We plot the MEAN epistemic uncertainty across all OOF observations for each chemical
    - More observations for a chemical → better-constrained chemical latent factors →
      lower contribution to epistemic uncertainty from the chemical side
    - Expected: NEGATIVE correlation (more data → lower uncertainty)
    
    Panel B: Aleatoric Uncertainty  
    - Aleatoric uncertainty (αc) is learned PER-CHEMICAL (constant across species)
    - Represents the model's posterior estimate of residual variance for that chemical
    - The posterior for αc depends on the residual variance observed for that chemical
    - With more observations, the posterior is more data-driven (less prior-influenced)
    - Negative correlation may indicate: (1) well-tested chemicals have standardized
      protocols yielding lower noise, (2) prior is conservative and more data reveals
      true lower variance, or (3) selection bias in which chemicals are heavily tested
    
    Args:
        df: DataFrame from load_oof_data() with columns including CAS, duration,
            epistemic_sd, aleatoric_sd
    """
    print("\n" + "="*60)
    print("FIGURE 1: Uncertainty vs Number of Observations (OOF)")
    print("="*60)
    
    # Count TOTAL observations per chemical (all durations), since alpha_c
    # is learned from all data for that chemical, not just one duration
    total_obs_per_chem = df.groupby("CAS", observed=True).size().rename("n_obs_total")
    
    # Filter to target duration for uncertainty values
    df_48h = df[df["duration"].astype(int) == DURATION_HOURS].copy()
    
    # Compute per-chemical uncertainty stats from OOF data at 48h
    # Epistemic varies by observation (depends on species + chemical), so take mean
    # Aleatoric: use mean, NOT first — OOF values differ across CV folds because
    # different folds learn different alpha_c from different training subsets
    chem_stats = df_48h.groupby("CAS", observed=True).agg(
        epistemic_sd=("epistemic_sd", "mean"),
        aleatoric_sd=("aleatoric_sd", "mean"),
        chem_name=("chem_name", "first")
    ).reset_index()
    
    # Merge with total observation counts (all durations)
    chem_stats = chem_stats.merge(total_obs_per_chem, on="CAS", how="left")
    
    # Exclude cold-start chemicals: those with fewer total observations than the
    # number of CV folds (3). These chemicals have folds where zero training data
    # exists for that CAS, so Gibbs never updates their CAS-specific parameters,
    # producing artificially low epistemic uncertainty.
    N_FOLDS = 3
    n_before = len(chem_stats)
    chem_stats = chem_stats[chem_stats["n_obs_total"] >= N_FOLDS].copy()
    n_excluded = n_before - len(chem_stats)
    
    print(f"Chemicals at {DURATION_HOURS}h: {n_before}")
    print(f"Excluded {n_excluded} cold-start chemicals (< {N_FOLDS} total obs)")
    print(f"Remaining: {len(chem_stats)}")
    print(f"Total observation range: {chem_stats['n_obs_total'].min()} - {chem_stats['n_obs_total'].max()}")
    
    # Calculate correlations
    rho_epist, p_epist = spearmanr(chem_stats["n_obs_total"], chem_stats["epistemic_sd"])
    rho_aleat, p_aleat = spearmanr(chem_stats["n_obs_total"], chem_stats["aleatoric_sd"])
    
    print(f"\nSpearman correlations:")
    print(f"  Epistemic SD vs # obs: ρ = {rho_epist:.3f} (p = {p_epist:.2e})")
    print(f"  Aleatoric SD vs # obs: ρ = {rho_aleat:.3f} (p = {p_aleat:.2e})")
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel A: Epistemic Uncertainty ---
    ax1 = axes[0]
    ax1.scatter(chem_stats["n_obs_total"], chem_stats["epistemic_sd"], 
                c='#377eb8', s=25, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Total Observations per Chemical (log scale)", fontsize=12)
    ax1.set_ylabel("Mean Epistemic SD (log mg/L)", fontsize=12)
    ax1.set_title(f"(a) Epistemic Uncertainty vs Data Availability\n"
                  f"Spearman ρ = {rho_epist:.3f}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Aleatoric Uncertainty ---
    ax2 = axes[1]
    ax2.scatter(chem_stats["n_obs_total"], chem_stats["aleatoric_sd"], 
                c='#e41a1c', s=25, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel("Total Observations per Chemical (log scale)", fontsize=12)
    ax2.set_ylabel("Aleatoric SD (log mg/L)", fontsize=12)
    ax2.set_title(f"(b) Aleatoric Uncertainty vs Data Availability\n"
                  f"Spearman ρ = {rho_aleat:.3f}", fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"uncertainty_vs_observations_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return chem_stats


def plot_example_predictions(df):
    """
    Plot predictions with uncertainty bars vs actual observations for 5 selected chemicals.
    
    For each chemical:
    - For each species tested, show predicted toxicity with uncertainty bars
    - Overlay actual measured values (aggregated by species)
    - This shows how well predictions align with reality and how uncertainty
      relates to prediction quality
    """
    print("\n" + "="*60)
    print("FIGURE 2: Example Predictions with Uncertainty")
    print("="*60)
    
    np.random.seed(RANDOM_SEED)
    
    # Filter to 48h duration
    df_48h = df[df["duration"].astype(int) == DURATION_HOURS].copy()
    
    # Count observations per chemical
    chem_counts = df_48h.groupby("CAS", observed=True).agg({
        "y_true": "count",
        "chem_name": "first"
    }).reset_index()
    chem_counts.columns = ["CAS", "n_obs", "chem_name"]
    
    # Filter to chemicals with sufficient observations
    eligible = chem_counts[chem_counts["n_obs"] >= MIN_OBS_FOR_EXAMPLES]
    print(f"Chemicals with ≥{MIN_OBS_FOR_EXAMPLES} observations at {DURATION_HOURS}h: {len(eligible)}")
    
    if len(eligible) < N_EXAMPLE_CHEMICALS:
        print(f"Warning: Only {len(eligible)} eligible chemicals, using all of them")
        selected = eligible
    else:
        # Select 5 random chemicals
        selected = eligible.sample(n=N_EXAMPLE_CHEMICALS, random_state=RANDOM_SEED)
    
    print(f"\nSelected chemicals:")
    for _, row in selected.iterrows():
        print(f"  {row['chem_name']} (CAS {row['CAS']}): {row['n_obs']} observations")
    
    # Create figure with subplots
    fig, axes = plt.subplots(N_EXAMPLE_CHEMICALS, 1, figsize=(12, 3.5 * N_EXAMPLE_CHEMICALS))
    if N_EXAMPLE_CHEMICALS == 1:
        axes = [axes]
    
    for idx, (_, chem_row) in enumerate(selected.iterrows()):
        ax = axes[idx]
        cas = chem_row["CAS"]
        chem_name = chem_row["chem_name"]
        
        # Get data for this chemical
        chem_data = df_48h[df_48h["CAS"] == cas].copy()
        
        # Aggregate by species: mean of y_true, y_pred, and uncertainties
        species_agg = chem_data.groupby("species", observed=True).agg({
            "y_true": ["mean", "std", "count"],
            "y_pred": "mean",
            "aleatoric_sd": "mean",
            "epistemic_sd": "mean",
            "total_sd": "mean"
        }).reset_index()
        
        # Flatten column names
        species_agg.columns = ["species", "y_true_mean", "y_true_std", "n_obs",
                               "y_pred", "aleatoric_sd", "epistemic_sd", "total_sd"]
        
        # Sort by predicted toxicity for cleaner visualization
        species_agg = species_agg.sort_values("y_pred").reset_index(drop=True)
        
        n_species = len(species_agg)
        y_positions = np.arange(n_species)
        
        # Plot decomposed uncertainty around predictions:
        # Outer bar (lighter): total uncertainty = epistemic + aleatoric
        ax.errorbar(species_agg["y_pred"], y_positions, 
                   xerr=1.96 * species_agg["total_sd"],
                   fmt='none', ecolor='#a6cee3', elinewidth=3, capsize=0, alpha=0.8,
                   label='Aleatoric component')
        
        # Inner bar (darker): epistemic uncertainty only
        ax.errorbar(species_agg["y_pred"], y_positions, 
                   xerr=1.96 * species_agg["epistemic_sd"],
                   fmt='o', color='#377eb8', markersize=5, 
                   ecolor='#377eb8', elinewidth=2, capsize=0, alpha=0.8,
                   label='Epistemic component')
        
        # Overlay actual observations
        ax.scatter(species_agg["y_true_mean"], y_positions, 
                  c='#e41a1c', s=50, marker='x', linewidths=2, zorder=5,
                  label='Observed (mean)')
        
        # Add observation variability where we have replicates (±1.96 SD for consistency)
        has_replicates = species_agg["n_obs"] > 1
        if has_replicates.any():
            ax.errorbar(species_agg.loc[has_replicates, "y_true_mean"],
                       y_positions[has_replicates],
                       xerr=1.96 * species_agg.loc[has_replicates, "y_true_std"],
                       fmt='none', ecolor='#e41a1c', elinewidth=1, capsize=2, alpha=0.5,
                       label='Observed ± 1.96 SD')
        
        # Formatting
        ax.set_xlabel("Toxicity (log mg/L)", fontsize=11)
        ax.set_ylabel("Species (sorted by predicted toxicity)", fontsize=11)
        ax.set_title(f"{chem_name} ({n_species} species tested at {DURATION_HOURS}h)", 
                    fontsize=12, fontweight='bold')
        ax.set_yticks([])  # Hide species names for cleanliness
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='lower right', fontsize=9)
        
        # Add counts annotation
        ax.annotate(f"N = {chem_row['n_obs']} observations\nacross {n_species} species",
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=9, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f"Model Predictions vs Observed Toxicity ({DURATION_HOURS}h exposure)\n"
                 "Dark blue = Epistemic, Light blue = Aleatoric, Red = Observed",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"example_predictions_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("GENERATING UNCERTAINTY CALIBRATION FIGURES")
    print("="*60)
    
    # Load OOF data (used by both figures)
    df = load_oof_data()
    
    # Figure 1: Uncertainty vs observation count (per chemical)
    plot_uncertainty_vs_observations(df)
    
    # Figure 2: Example predictions with uncertainty bars
    plot_example_predictions(df)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

