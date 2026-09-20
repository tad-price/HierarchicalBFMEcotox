"""
dataset_figures.py - Dataset Characterization Figures and Tables

Generates:
- dataset_summary.csv: summary statistics of the ADORE dataset
- replicate_calibration.csv: aleatoric SD against empirical replicate SD, by replicate count
- rank_freq.png: rank-frequency of observations per species and per chemical
- replicate_sd_distribution_all_durations.png: within-triplet replicate SDs

Requires:
- data/raw/ecotox_mortality_processed.csv
- data/raw/ecotox_properties_with-oecd-function.csv
- outputs/models/oof_aleatoric.npy (for Table 2)
- outputs/models/oof_mean.npy (for Table 2)

Writes to outputs/figures/.

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
# Dataset summary statistics
# =============================================================================

def dataset_summary_table(df):
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
    print("Summary statistics of the ADORE dataset")
    print("=" * 60)
    for label, val in rows:
        print(f"  {label:<60s} {val}")

    pd.DataFrame(rows, columns=["Statistic", "Value"]).to_csv(
        OUTPUT_DIR / "dataset_summary.csv", index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'dataset_summary.csv'}")


# =============================================================================
# Aleatoric calibration by replicate count
# =============================================================================

def _null_median_ratio(pred_sd, n_reps, n_sim=200, seed=0):
    """
    Median predicted-to-empirical SD ratio expected under perfect calibration.

    Treats each triplet's predicted SD as the true noise level, draws that
    triplet's actual number of replicates from a Gaussian with that SD, and
    recomputes the ratio. The result is the reference value the observed median
    ratio should be read against: it is above one even for a perfectly
    calibrated model, because the empirical SD in the denominator is estimated
    from few replicates and is both noisy and biased low.
    """
    rng = np.random.default_rng(seed)
    pred_sd = np.asarray(pred_sd, dtype=float)
    n_reps = np.asarray(n_reps, dtype=int)
    medians = np.empty(n_sim)
    for j in range(n_sim):
        emp = np.array([sd * rng.standard_normal(n).std(ddof=1)
                        for sd, n in zip(pred_sd, n_reps)])
        medians[j] = np.median(pred_sd / emp)
    return medians.mean()


def replicate_calibration_table(df):
    """
    Predicted vs empirical SD by replicate-count bin.

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
    n_degenerate = 0
    for _, g in grouped:
        n = len(g)
        if n < 5:
            continue
        emp_sd = g["y_true"].std()
        if emp_sd < 1e-6:  # duplicate/identical records; useless for calibration
            n_degenerate += 1
            continue
        pred_sd = np.sqrt(g["pred_aleatoric_var"].mean())
        records.append({"n": n, "emp_sd": emp_sd, "pred_sd": pred_sd})

    recs = pd.DataFrame(records)
    if n_degenerate:
        print(f"Dropped {n_degenerate} degenerate triplet(s) with emp_sd < 1e-6")

    bins = [(5, 9), (10, 19), (20, 49), (50, 99), (100, None)]
    bin_labels = ["5-9", "10-19", "20-49", "50-99", "100+"]

    print("\n" + "=" * 60)
    print("Aleatoric calibration by replicate count")
    print("=" * 60)
    header = (f"{'Replicates':<12} {'N groups':<10} {'Pooled Ratio':<14} "
              f"{'Median Ratio':<14} {'Null Median':<13} {'Ratio IQR':<18} "
              f"{'Mean Pred SD':<14} {'Mean Emp SD':<12}")
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
        ratio = sub["pred_sd"] / sub["emp_sd"]
        pooled = sub["pred_sd"].mean() / sub["emp_sd"].mean()
        q25, q75 = ratio.quantile(0.25), ratio.quantile(0.75)
        null_med = _null_median_ratio(sub["pred_sd"].values, sub["n"].values)
        row = {
            "Replicates": label,
            "N groups": len(sub),
            "Pooled Ratio": f"{pooled:.3f}",
            "Median Ratio": f"{ratio.median():.3f}",
            "Null Median Ratio": f"{null_med:.3f}",
            "Ratio IQR": f"[{q25:.3f}, {q75:.3f}]",
            "Mean Pred SD": f"{sub['pred_sd'].mean():.3f}",
            "Mean Emp SD": f"{sub['emp_sd'].mean():.3f}",
        }
        table_rows.append(row)
        print(f"{label:<12} {len(sub):<10} {pooled:<14.3f} {ratio.median():<14.3f} "
              f"{null_med:<13.3f} "
              f"{'['+format(q25,'.3f')+', '+format(q75,'.3f')+']':<18} "
              f"{sub['pred_sd'].mean():<14.3f} {sub['emp_sd'].mean():<12.3f}")

    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "replicate_calibration.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'replicate_calibration.csv'}")


# =============================================================================
# Rank-frequency plots
# =============================================================================

def plot_rank_frequency(df):
    """
    Rank-frequency plots for species and chemicals.

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
# Replicate-SD distribution
# =============================================================================

def plot_replicate_sd_distribution(df):
    """
    Distribution of the within-triplet standard deviation of
    log10 LC50 for (species, CAS, duration) groups with at least 10 observations.

    An earlier version of this figure plotted a relative standard deviation,
    std(conc) / |mean(conc)|. Because `conc` is already log10-transformed, that
    ratio diverges whenever the mean log LC50 approaches zero (LC50 near
    1 mg/L) and is therefore driven by the denominator rather than by the
    spread of the replicates. The standard deviation of log10 LC50 is reported
    instead: it is stable, and 10**SD reads directly as the multiplicative
    factor separating typical repeat tests (the geometric standard deviation).
    """
    grouped = df.groupby(["species", "CAS", "duration"], observed=True)["conc"]
    stats = grouped.agg(["mean", "std", "count"])
    stats = stats[stats["count"] >= 10].copy()
    stats = stats.dropna(subset=["std"])

    sd = stats["std"]
    median_sd = sd.median()
    p90_sd = sd.quantile(0.90)
    print(f"  {len(sd)} triplets with >=10 replicates; "
          f"median SD = {median_sd:.3f} (GSD {10 ** median_sd:.2f}x), "
          f"p90 SD = {p90_sd:.3f} (GSD {10 ** p90_sd:.2f}x)")

    # Deliberately spare: no title and minimal annotation, since the details
    # belong in the manuscript caption rather than on the figure itself.
    x_max = 2.0
    n_beyond = int((sd > x_max).sum())
    print(f"  {n_beyond} triplets beyond the displayed range "
          f"(SD > {x_max:.1f}); quote this in the caption")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(sd, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(median_sd, color='#a03a20', linestyle='--', linewidth=1.5,
               label="median")
    ax.axvline(p90_sd, color='#8a6410', linestyle=':', linewidth=1.5,
               label="90th percentile")
    ax.set_xlabel("Within-triplet SD of log$_{10}$ LC50 (log mg/L)", fontsize=12)
    ax.set_ylabel("Number of triplets", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "replicate_sd_distribution_all_durations.png"
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

    dataset_summary_table(df)
    plot_rank_frequency(df)
    plot_replicate_sd_distribution(df)

    # The replicate-calibration table needs the out-of-fold arrays
    oof_path = ROOT_DIR / "outputs" / "models" / "oof_mean.npy"
    if oof_path.exists():
        replicate_calibration_table(df)
    else:
        print("\nSkipping the replicate-calibration table: no out-of-fold arrays. "
              "Run scripts/train_bfm.py first.")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
