"""
dataset_figures.py - Dataset Characterization Figures and Tables

Generates:
- Table 1: Summary statistics of the ADORE dataset (printed + CSV)
- Table 2: Aleatoric calibration by replicate count (printed + CSV)
- Figure 1: Rank-frequency plots for species and chemicals
- Figure 2: Distribution of RSDs grouped by species, chemical, duration

Requires:
- data/raw/ecotox_mortality_processed.csv
- data/raw/ecotox_properties_with-oecd-function.csv
- outputs/models/oof_aleatoric.npy (for Table 2)
- outputs/models/oof_mean.npy (for Table 2)

Outputs to:
- outputs/figures/dataset/

Usage:
    python analysis/dataset_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data

OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DURATION_HOURS = 48


def load_data():
    """Load the ADORE ecotox dataset."""
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42,
    )
    full_data["y_true"] = y_centered + y_mean
    return full_data


# =============================================================================
# TABLE 1: Dataset summary statistics
# =============================================================================

def table1_summary(df):
    """Print and save Table 1: summary statistics of the ADORE dataset."""
    n_chemicals = df["CAS"].nunique()
    n_species = df["species"].nunique()
    n_durations = df["duration"].nunique()

    n_pairs = df.groupby(["species", "CAS"], observed=True).ngroups
    triplets = df.groupby(["species", "CAS", "duration"], observed=True).ngroups
    n_obs = len(df)

    sparsity = 1.0 - n_pairs / (n_chemicals * n_species)

    rows = [
        ("Number of unique chemicals", n_chemicals),
        ("Number of unique species", n_species),
        ("Number of durations", n_durations),
        ("Number of unique (species, chemical, duration) triplets observed", triplets),
        ("Total number of observations", n_obs),
        ("Sparsity (species*chemical matrix)", f"{sparsity:.1%}"),
    ]

    print("\n" + "=" * 60)
    print("TABLE 1: Summary statistics of the ADORE dataset")
    print("=" * 60)
    for label, val in rows:
        print(f"  {label:<60s} {val}")

    pd.DataFrame(rows, columns=["Statistic", "Value"]).to_csv(
        OUTPUT_DIR / "table1_summary.csv", index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'table1_summary.csv'}")


# =============================================================================
# TABLE 2: Aleatoric calibration by replicate count
# =============================================================================

def table2_aleatoric_calibration(df):
    """
    Print and save Table 2: predicted vs empirical SD by replicate bin.

    Requires OOF predictions (oof_mean.npy, oof_aleatoric.npy) to exist.
    """
    models_dir = ROOT_DIR / "outputs" / "models"
    oof_mean = np.load(models_dir / "oof_mean.npy")
    oof_aleatoric = np.load(models_dir / "oof_aleatoric.npy")

    df = df.copy()
    df["pred_mean"] = oof_mean
    df["pred_aleatoric_var"] = oof_aleatoric

    # Group by (species, CAS, duration) triplet
    grouped = df.groupby(["species", "CAS", "duration"], observed=True)

    records = []
    for _, g in grouped:
        n = len(g)
        if n < 5:
            continue
        emp_sd = g["y_true"].std()
        pred_sd = np.sqrt(g["pred_aleatoric_var"].mean())
        records.append({"n": n, "emp_sd": emp_sd, "pred_sd": pred_sd})

    recs = pd.DataFrame(records)

    bins = [(5, 9), (10, 19), (20, 49), (50, 99), (100, None)]
    bin_labels = ["5-9", "10-19", "20-49", "50-99", "100+"]

    print("\n" + "=" * 60)
    print("TABLE 2: Aleatoric calibration by replicate count")
    print("=" * 60)
    header = f"{'Replicates':<12} {'N groups':<10} {'Mean Ratio':<12} {'Median Ratio':<14} {'Mean Pred SD':<14} {'Mean Emp SD':<12}"
    print(header)
    print("-" * len(header))

    table_rows = []
    for (lo, hi), label in zip(bins, bin_labels):
        if hi is not None:
            mask = (recs["n"] >= lo) & (recs["n"] <= hi)
        else:
            mask = recs["n"] >= lo
        sub = recs[mask]
        if len(sub) == 0:
            continue
        ratio = sub["pred_sd"] / sub["emp_sd"].replace(0, np.nan)
        row = {
            "Replicates": label,
            "N groups": len(sub),
            "Mean Ratio": f"{ratio.mean():.3f}",
            "Median Ratio": f"{ratio.median():.3f}",
            "Mean Pred SD": f"{sub['pred_sd'].mean():.3f}",
            "Mean Emp SD": f"{sub['emp_sd'].mean():.3f}",
        }
        table_rows.append(row)
        print(f"{label:<12} {len(sub):<10} {ratio.mean():<12.3f} {ratio.median():<14.3f} {sub['pred_sd'].mean():<14.3f} {sub['emp_sd'].mean():<12.3f}")

    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "table2_aleatoric_calibration.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'table2_aleatoric_calibration.csv'}")


# =============================================================================
# FIGURE 1: Rank-frequency plots
# =============================================================================

def figure1_rank_frequency(df):
    """
    Generate Figure 1: rank-frequency plots for species and chemicals.

    Left panel: observations per species (ranked).
    Right panel: observations per chemical (ranked).
    """
    species_counts = df.groupby("species", observed=True).size().sort_values(ascending=False).values
    chemical_counts = df.groupby("CAS", observed=True).size().sort_values(ascending=False).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, len(species_counts) + 1), species_counts, linewidth=1.5, color='steelblue')
    ax1.set_yscale("log")
    ax1.set_xlabel("Species (ranked)", fontsize=12)
    ax1.set_ylabel("Number of Observations", fontsize=12)
    ax1.set_title("Rank-Frequency Plot for Species", fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(chemical_counts) + 1), chemical_counts, linewidth=1.5, color='steelblue')
    ax2.set_yscale("log")
    ax2.set_xlabel("Chemicals (ranked)", fontsize=12)
    ax2.set_ylabel("Number of Observations", fontsize=12)
    ax2.set_title("Rank-Frequency Plot for chemicals", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "rank_freq.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 2: RSD distribution
# =============================================================================

def figure2_rsd_distribution(df):
    """
    Generate Figure 2: distribution of relative standard deviations (RSDs)
    for (species, CAS, duration) groups with at least 10 observations.

    RSD = std(conc) / |mean(conc)|, computed on the raw (non-centered) values.
    """
    grouped = df.groupby(["species", "CAS", "duration"], observed=True)["conc"]
    stats = grouped.agg(["mean", "std", "count"])
    stats = stats[stats["count"] >= 10].copy()
    stats["rsd"] = stats["std"] / stats["mean"].abs()
    stats = stats.dropna(subset=["rsd"])
    stats = stats[stats["rsd"].between(0, 8)]  # trim extreme outliers for display

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stats["rsd"], bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel("RSD of conc", fontsize=12)
    ax.set_ylabel("Count of (species, CAS) pairs", fontsize=12)
    ax.set_title("Distribution of Relative Standard Deviations\n(all groups, n > 10)", fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "RSD_distribution_all_durations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DATASET CHARACTERIZATION")
    print("=" * 60)

    df = load_data()

    table1_summary(df)
    figure1_rank_frequency(df)
    figure2_rsd_distribution(df)

    # Table 2 requires OOF predictions; skip gracefully if not available
    oof_path = ROOT_DIR / "outputs" / "models" / "oof_mean.npy"
    if oof_path.exists():
        table2_aleatoric_calibration(df)
    else:
        print("\nSkipping Table 2 (OOF predictions not found). Run scripts/train_bfm.py first.")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
