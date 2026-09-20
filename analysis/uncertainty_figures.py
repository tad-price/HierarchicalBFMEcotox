"""
uncertainty_figures.py - Generate Uncertainty Calibration Figures for Paper

This script generates two key figures for the Uncertainty Calibration section:

1. Uncertainty vs Observation Count: Shows how epistemic and aleatoric uncertainty
   relate to the amount of data per chemical. Epistemic should decrease with more
   data (model confidence increases); aleatoric is the learned irreducible noise.

2. Example Predictions with Uncertainty: For the selected chemicals, shows model
   predictions with uncertainty bars compared to actual observed measurements.

Usage:
    python analysis/uncertainty_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model
from scipy.stats import spearmanr

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data

# Configuration
DURATION_HOURS = 48
CV_N_FOLDS = 5  # must match the --n_folds used to produce the saved OOF arrays
MIN_OBS_FOR_EXAMPLES = 10  # Minimum observations for example chemicals
N_EXAMPLE_CHEMICALS = 4
# Excluded from the example panel: too few species over too narrow a toxicity
# range for the panel to show any contrast between predictions.
EXCLUDE_FROM_EXAMPLES = ["2,6-Dimethylquinoline"]
RANDOM_SEED = 42

OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def cold_start_chemicals(df, n_folds=CV_N_FOLDS):
    """
    Return the CAS codes that are absent from the training partition of at least
    one cross-validation fold.

    Reproduces the grouped split of scripts/train_bfm.py: GroupKFold on the
    (CAS, species, duration) triplet identifier. A chemical whose rows all land
    in one held-out fold is never seen by that fold's sampler, so its
    CAS-specific parameters stay at the prior and its epistemic variance is
    deflated. Such chemicals carry no information about the relationship between
    uncertainty and data availability and are dropped from the per-chemical
    analyses.

    This tests the condition directly. The earlier rule of thumb (drop chemicals
    with fewer observations than there are folds) both kept chemicals that are
    genuinely cold-start and dropped chemicals that are not.
    """
    d = df.copy()
    d["duration"] = pd.Categorical(d["duration"].astype(int))
    triplet_id = pd.factorize(
        d["CAS"].astype(str) + "_"
        + d["species"].astype(str) + "_"
        + d["duration"].astype(str)
    )[0]
    cas = df["CAS"].values
    cold = set()
    for tr_idx, va_idx in sk_model.GroupKFold(n_splits=n_folds).split(d, groups=triplet_id):
        cold |= set(cas[va_idx]) - set(cas[tr_idx])
    return cold


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


def plot_uncertainty_vs_observations(df):
    """
    Mean epistemic and aleatoric SD per chemical against that chemical's
    observation count, one panel each, with Spearman correlations.

    Uses out-of-fold predictions, so each observation's uncertainty comes from a
    model that did not train on it. Epistemic SD is averaged over the chemical's
    observations at the target duration; aleatoric SD is averaged too, since the
    out-of-fold value differs between folds.

    Args:
        df: DataFrame from load_oof_data() with columns including CAS, duration,
            epistemic_sd, aleatoric_sd
    """
    print("\n" + "="*60)
    print("Uncertainty vs number of observations (out-of-fold)")
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
    
    # Exclude cold-start chemicals: those absent from the training partition of
    # at least one fold, so that Gibbs never updates their CAS-specific
    # parameters and their epistemic uncertainty comes out artificially low.
    cold = cold_start_chemicals(df)
    n_before = len(chem_stats)
    chem_stats = chem_stats[~chem_stats["CAS"].isin(cold)].copy()
    n_excluded = n_before - len(chem_stats)
    
    print(f"Chemicals at {DURATION_HOURS}h: {n_before}")
    print(f"Excluded {n_excluded} cold-start chemicals "
          f"(absent from the training partition of >=1 of {CV_N_FOLDS} folds)")
    print(f"Remaining: {len(chem_stats)}")
    print(f"Total observation range: {chem_stats['n_obs_total'].min()} - {chem_stats['n_obs_total'].max()}")
    
    # Calculate correlations
    rho_epist, p_epist = spearmanr(chem_stats["n_obs_total"], chem_stats["epistemic_sd"])
    rho_aleat, p_aleat = spearmanr(chem_stats["n_obs_total"], chem_stats["aleatoric_sd"])
    
    print("\nSpearman correlations:")
    print(f"  Epistemic SD vs # obs: ρ = {rho_epist:.3f} (p = {p_epist:.2e})")
    print(f"  Aleatoric SD vs # obs: ρ = {rho_aleat:.3f} (p = {p_aleat:.2e})")
    
    # Create figure with two panels
    # No in-panel titles: the (a)/(b) mapping and the Spearman rho are carried by
    # the manuscript caption and body text respectively. The rho values are still
    # printed above, which is where they should be read from when updating them.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # --- Panel A: Epistemic Uncertainty ---
    ax1 = axes[0]
    ax1.scatter(chem_stats["n_obs_total"], chem_stats["epistemic_sd"], 
                c='#377eb8', s=25, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Total Observations per Chemical (log scale)", fontsize=12)
    ax1.set_ylabel("Mean epistemic SD (log mg/L)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Aleatoric Uncertainty ---
    ax2 = axes[1]
    ax2.scatter(chem_stats["n_obs_total"], chem_stats["aleatoric_sd"], 
                c='#e41a1c', s=25, alpha=0.6, edgecolors='white', linewidths=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel("Total Observations per Chemical (log scale)", fontsize=12)
    ax2.set_ylabel("Mean aleatoric SD (log mg/L)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"uncertainty_vs_observations_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return chem_stats


def plot_example_predictions(df):
    """
    Plot predictions with uncertainty bars vs actual observations for the selected chemicals.
    
    For each chemical:
    - For each species tested, show predicted toxicity with uncertainty bars
    - Overlay actual measured values (aggregated by species)
    - This shows how well predictions align with reality and how uncertainty
      relates to prediction quality
    """
    print("\n" + "="*60)
    print("Example predictions with uncertainty")
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
    
    # Draw the excluded chemicals as well, then drop them, so that the retained
    # panels do not depend on how many are excluded
    n_draw = N_EXAMPLE_CHEMICALS + len(EXCLUDE_FROM_EXAMPLES)
    if len(eligible) < n_draw:
        print(f"Warning: Only {len(eligible)} eligible chemicals, using all of them")
        selected = eligible
    else:
        selected = eligible.sample(n=n_draw, random_state=RANDOM_SEED)
    selected = selected[~selected["chem_name"].isin(EXCLUDE_FROM_EXAMPLES)]
    selected = selected.head(N_EXAMPLE_CHEMICALS)

    print("\nSelected chemicals:")
    for _, row in selected.iterrows():
        print(f"  {row['chem_name']} (CAS {row['CAS']}): {row['n_obs']} observations")

    # Create figure with subplots
    n_panels = len(selected)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels))
    if n_panels == 1:
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
        # Outer bar (lighter): total uncertainty = epistemic + aleatoric.
        # Labelled as the total, not as the aleatoric component: the aleatoric
        # part is the width this bar adds to the inner one, not the bar itself.
        ax.errorbar(species_agg["y_pred"], y_positions, 
                   xerr=1.96 * species_agg["total_sd"],
                   fmt='none', ecolor='#a6cee3', elinewidth=3, capsize=0, alpha=0.8,
                   label='Total: epistemic + aleatoric (±1.96σ)')
        
        # Inner bar (darker): epistemic uncertainty only
        ax.errorbar(species_agg["y_pred"], y_positions, 
                   xerr=1.96 * species_agg["epistemic_sd"],
                   fmt='o', color='#377eb8', markersize=5, 
                   ecolor='#377eb8', elinewidth=2, capsize=0, alpha=0.8,
                   label='Epistemic only (±1.96σ)')
        
        # Overlay actual observations
        ax.scatter(species_agg["y_true_mean"], y_positions, 
                  c='#e41a1c', s=50, marker='x', linewidths=2, zorder=5,
                  label='Observed (mean)')
        
        # Add observation variability where replicates exist (±1.96 SD)
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
                 "Dark blue = epistemic, light blue = total, red = observed",
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
    
    plot_uncertainty_vs_observations(df)
    
    plot_example_predictions(df)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

