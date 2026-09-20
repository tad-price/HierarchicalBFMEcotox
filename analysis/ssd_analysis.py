"""Per-species SSD with decomposed uncertainty, for one chemical and duration.

Plots every species prediction for the target chemical against its rank,
with error bars split into an epistemic band and a total band whose extra
width is the per-chemical aleatoric term.

The posterior-ensemble SSD, which carries uncertainty on the curve as a whole
rather than on the individual species, is in ssd_mc_uncertainty.py.

Requires:
- outputs/models/full_predictions.parquet

Writes <chemical>_novel_uncertainty.png to outputs/figures/.

Usage:
    python analysis/ssd_analysis.py --cas 1912-24-9
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def plot_novel_ssd_with_uncertainty(pred_df, z_score=1.96):
    """
    Generate a novel SSD scatterplot with decomposed uncertainty bars.
    
    Methodology:
    - For each species, plot the predicted toxicity as a point
    - Add horizontal error bars showing:
      1. Inner bar (darker): Epistemic uncertainty (model uncertainty, varies per species)
      2. Outer bar (lighter): Total uncertainty (epistemic + aleatoric)
    - Uses ±z_score standard deviations (default 1.96 for 95% CI)

    The inner/outer ordering matches the example-prediction figure produced by
    uncertainty_figures.py, so that the darker bar always means the same thing:
    the species-specific model uncertainty.
    
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
    epistemic_err = z_score * epistemic_sd
    total_err = z_score * total_sd  # Total includes both components

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # With ~1,200 species the individual error bars sit closer together than the
    # line width, so they are drawn as nested envelopes rather than as separate
    # bars. Each y-position still carries its own species-specific width. The
    # envelope edges are smoothed with a rolling median across adjacent ranks
    # purely for legibility; the plotted points and the reported SDs are not
    # smoothed.
    smooth_window = max(1, n_species // 50)

    def _smooth(values):
        if smooth_window <= 1:
            return values
        return (pd.Series(values)
                .rolling(smooth_window, center=True, min_periods=1)
                .median()
                .values)

    ax.fill_betweenx(y_cdf, _smooth(means - total_err), _smooth(means + total_err),
                     color='#6baed6', alpha=0.75, linewidth=0,
                     label=f'Total: epistemic + aleatoric (±{z_score:.2f}σ)')
    ax.fill_betweenx(y_cdf, _smooth(means - epistemic_err), _smooth(means + epistemic_err),
                     color='#08519c', alpha=0.75, linewidth=0,
                     label=f'Epistemic only (±{z_score:.2f}σ)')

    # Median prediction curve on top
    ax.plot(means, y_cdf, color='#08306b', linewidth=1.8, zorder=5,
            label='Predicted SSD (posterior mean)')

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
    print("SPECIES SENSITIVITY DISTRIBUTION ANALYSIS")
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

    plot_novel_ssd_with_uncertainty(pred_df_filtered)

    print("\n" + "="*60)
    print(f"Chemical: {CHEMICAL_NAME}")
    print(f"Duration: {DURATION_HOURS}h")
    print(f"Observed species: {df_obs_filtered['species'].nunique()}")
    print(f"Predicted species: {len(pred_df_filtered)}")
    print("="*60)


if __name__ == "__main__":
    main()
