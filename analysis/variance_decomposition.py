"""
variance_decomposition.py - Aleatoric vs Epistemic Variance Decomposition

Quantifies how the model's predictive variance splits into aleatoric (per-chemical
irreducible noise, 1/alpha_c) and epistemic (posterior model uncertainty) components,
and how that split shifts as the number of observations per chemical grows.

Outputs:
- Console: global aleatoric share of total predictive variance.
- outputs/figures/variance_share_by_nobs.csv: bucketed decomposition table.
- outputs/figures/aleatoric_epistemic_ratio_vs_nobs.png: ratio vs observations/chemical.

Uses out-of-fold predictions. Chemicals absent from the training partition of at
least one fold ("cold-start") have their chemical-specific parameters left at the
prior in that fold, which deflates their epistemic variance, and are excluded by
uncertainty_figures.cold_start_chemicals().

Usage:
    python analysis/variance_decomposition.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.load_ecotox import load_ecotox_data
from uncertainty_figures import cold_start_chemicals

OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

BUCKETS = [(3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 249), (250, None)]
BUCKET_LABELS = ["3-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]


def load():
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False, use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42)
    models = ROOT_DIR / "outputs" / "models"
    df = full_data.copy()
    df["aleatoric_var"] = np.load(models / "oof_aleatoric.npy")
    df["epistemic_var"] = np.load(models / "oof_epistemic.npy")
    df["total_var"] = df["aleatoric_var"] + df["epistemic_var"]
    return df


def plot_variance_composition(cc):
    """Stacked absolute mean variance (aleatoric + epistemic) vs observations/chemical."""
    labels, n_chem, mean_ale, mean_epi = [], [], [], []
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        m = (cc["n_obs"] >= lo) & ((cc["n_obs"] <= hi) if hi else True)
        s = cc[m]
        if len(s) == 0:
            continue
        labels.append(lab); n_chem.append(len(s))
        mean_ale.append(s["aleatoric_var"].mean())
        mean_epi.append(s["epistemic_var"].mean())
    mean_ale = np.array(mean_ale); mean_epi = np.array(mean_epi)
    total = mean_ale + mean_epi
    x = np.arange(len(labels))

    C_ALE, C_EPI = "#d1495b", "#3b7dba"
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(x, mean_ale, color=C_ALE, label="Aleatoric (irreducible noise)")
    ax.bar(x, mean_epi, bottom=mean_ale, color=C_EPI, label="Epistemic (model uncertainty)")
    for xi, t, n in zip(x, total, n_chem):
        ax.annotate(f"{t:.2f}", (xi, t), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold")
        ax.annotate(f"n={n}", (xi, 0), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Observations per chemical", fontsize=12)
    ax.set_ylabel(r"Mean predictive variance  (log mg/L)$^2$", fontsize=12)
    ax.set_title("Total predictive variance and its composition", fontsize=13)
    ax.legend(loc="upper center", fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "variance_composition_vs_nobs.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    df = load()

    chem = df.groupby("CAS", observed=True).agg(
        aleatoric_var=("aleatoric_var", "mean"),
        epistemic_var=("epistemic_var", "mean"),
    )
    chem["n_obs"] = df.groupby("CAS", observed=True).size()
    chem["ratio"] = chem["aleatoric_var"] / chem["epistemic_var"]
    chem["ale_share"] = chem["aleatoric_var"] / (chem["aleatoric_var"] + chem["epistemic_var"])

    cold = cold_start_chemicals(df)
    keep = ~df["CAS"].isin(cold)
    a, e = df.loc[keep, "aleatoric_var"], df.loc[keep, "epistemic_var"]
    pooled = a.mean() / (a.mean() + e.mean())
    perobs = (a / (a + e))
    print("=" * 60)
    print("GLOBAL ALEATORIC SHARE OF TOTAL PREDICTIVE VARIANCE")
    print("=" * 60)
    print(f"  Pooled (sum aleatoric / sum total) : {pooled:.1%}")
    print(f"  Per-observation share: mean={perobs.mean():.1%}  median={perobs.median():.1%}")
    print(f"  (excluding {(~keep).sum()} rows from {len(cold)} cold-start chemicals)")

    cc = chem[~chem.index.isin(cold)].copy()
    print("\n" + "=" * 60)
    print("DECOMPOSITION BY OBSERVATIONS PER CHEMICAL")
    print("=" * 60)
    print(f"{'Obs/chem':>10} {'N chem':>7} {'Med Aleat':>10} {'Med Epist':>10} "
          f"{'Ale/Epi':>8} {'Ale share':>10}")
    rows = []
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        m = (cc["n_obs"] >= lo) & ((cc["n_obs"] <= hi) if hi else True)
        s = cc[m]
        if len(s) == 0:
            continue
        med_ratio = s["ratio"].median()
        med_share = s["ale_share"].median()
        rows.append({
            "Obs per chemical": lab, "N chemicals": len(s),
            "Median aleatoric var": round(s["aleatoric_var"].median(), 4),
            "Median epistemic var": round(s["epistemic_var"].median(), 4),
            "Median aleatoric/epistemic": round(med_ratio, 3),
            "Median aleatoric share": round(med_share, 4),
        })
        print(f"{lab:>10} {len(s):>7} {s['aleatoric_var'].median():>10.4f} "
              f"{s['epistemic_var'].median():>10.4f} {med_ratio:>8.2f} {med_share:>9.1%}")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "variance_share_by_nobs.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'variance_share_by_nobs.csv'}")

    plot_variance_composition(cc)

    bx = [np.sqrt(lo * (hi if hi else lo * 4)) for (lo, hi) in BUCKETS]  # geometric bucket centres
    by = [cc[(cc["n_obs"] >= lo) & ((cc["n_obs"] <= hi) if hi else True)]["ratio"].median()
          for (lo, hi) in BUCKETS]
    crossover = None
    for i in range(1, len(by)):
        if by[i - 1] < 1 <= by[i]:
            t = (np.log(1) - np.log(by[i - 1])) / (np.log(by[i]) - np.log(by[i - 1]))
            crossover = np.exp(np.log(bx[i - 1]) + t * (np.log(bx[i]) - np.log(bx[i - 1])))
            break
    if crossover:
        print(f"\nAleatoric overtakes epistemic (ratio=1) at "
              f"~{crossover:.0f} observations per chemical.")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cc["n_obs"], cc["ratio"], s=14, alpha=0.25, c="#888888",
               edgecolors="none", label="Per chemical")
    ax.plot(bx, by, "o-", color="#e41a1c", lw=2.2, ms=8, label="Bucket median")
    for x, y, lab in zip(bx, by, BUCKET_LABELS):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#e41a1c")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 50)  # clip extreme per-chemical ratios for readability
    ax.axhline(1.0, ls="--", color="black", lw=1)
    ax.text(0.98, 0.5, "aleatoric = epistemic", ha="right", va="bottom", fontsize=9,
            transform=ax.get_yaxis_transform())  # x in axes-frac, y in data coords
    if crossover:
        ax.axvline(crossover, ls=":", color="#377eb8", lw=1.5)
        ax.text(crossover, 0.02, f"  ~{crossover:.0f} obs", color="#377eb8",
                va="bottom", ha="left", fontsize=9,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("Total observations per chemical (log scale)", fontsize=12)
    ax.set_ylabel("Aleatoric / Epistemic variance ratio (log scale)", fontsize=12)
    ax.set_title("Aleatoric vs epistemic variance by data availability", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / "aleatoric_epistemic_ratio_vs_nobs.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
