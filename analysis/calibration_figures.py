"""
calibration_figures.py - Posterior Predictive Calibration

Validates whether the model's total predictive uncertainty is well-calibrated:
for a nominal X% credible interval, does the true value fall inside roughly
X% of the time?

For each out-of-fold observation we have S posterior samples
{ y_hat^(s), 1/alpha_c^(s) }. The posterior predictive distribution is the
mixture
    p(y* | x*, D) = (1/S) sum_s N( y_hat^(s)(x*), 1/alpha_c^(s) ).
We approximate it by drawing one Monte Carlo sample from each Gaussian
component, then take empirical quantiles to construct credible intervals at a
range of nominal levels. Coverage at level p is the fraction of OOF
observations whose true y falls inside the interval.

Requires:
- outputs/models/oof_pred_samples.npy        shape (N, S)
- outputs/models/oof_aleatoric_samples.npy   shape (N, S)

Outputs:
- outputs/figures/calibration_curve.png
- outputs/figures/calibration_curve.csv

Usage:
    python analysis/calibration_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.load_ecotox import load_ecotox_data

MODELS_DIR = ROOT_DIR / "outputs" / "models"
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NOMINAL_LEVELS = np.array([0.10, 0.20, 0.30, 0.40, 0.50,
                           0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
RANDOM_SEED = 42


def load_inputs():
    """Load OOF posterior samples and the corresponding true y values."""
    pred_path = MODELS_DIR / "oof_pred_samples.npy"
    alea_path = MODELS_DIR / "oof_aleatoric_samples.npy"
    if not pred_path.exists() or not alea_path.exists():
        print(f"Missing per-sample artifacts in {MODELS_DIR}.")
        print("Re-run scripts/train_bfm.py to generate "
              "oof_pred_samples.npy and oof_aleatoric_samples.npy.")
        sys.exit(1)

    pred_samples = np.load(pred_path)        # (N, S), centered scale
    alea_samples = np.load(alea_path)        # (N, S), variances
    print(f"Loaded posterior samples: shape {pred_samples.shape}")

    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42,
    )
    if len(y_centered) != pred_samples.shape[0]:
        raise ValueError(
            f"Length mismatch: y_centered={len(y_centered)} vs "
            f"pred_samples={pred_samples.shape[0]}. The dataset and the "
            "saved samples must come from the same training run."
        )
    return pred_samples, alea_samples, y_centered


def draw_posterior_predictive(pred_samples, alea_samples, rng):
    """
    Draw one sample from each posterior component to form the posterior
    predictive ensemble. Returns array of shape (N, S).

    pred_samples and alea_samples must already be on the same y-scale.
    """
    sd = np.sqrt(np.maximum(alea_samples, 0.0))
    noise = rng.standard_normal(pred_samples.shape).astype(pred_samples.dtype)
    return pred_samples + sd * noise


def coverage_at_levels(pp_draws, y_true, levels):
    """
    Empirical coverage of central credible intervals.

    pp_draws: (N, S) posterior predictive draws
    y_true:   (N,)
    levels:   1-D array of nominal coverage levels in (0, 1)

    For each level p we take the central interval [q_lo, q_hi] with
    q_lo = (1-p)/2 and q_hi = (1+p)/2 quantiles per row, then return the
    fraction of rows where y_true falls in [q_lo, q_hi].
    """
    lo_q = (1.0 - levels) / 2.0
    hi_q = (1.0 + levels) / 2.0
    # np.quantile across S for each observation, vectorized over levels
    q_lo = np.quantile(pp_draws, lo_q, axis=1).T   # (N, L)
    q_hi = np.quantile(pp_draws, hi_q, axis=1).T   # (N, L)
    inside = (y_true[:, None] >= q_lo) & (y_true[:, None] <= q_hi)
    return inside.mean(axis=0)


def plot_calibration(levels, empirical, n_obs, output_path):
    """Calibration plot: nominal vs empirical coverage with diagonal."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect-calibration reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7,
            label="Perfect calibration")

    # Model curve
    ax.plot(levels, empirical, "o-", color="#377eb8", linewidth=2,
            markersize=7, label="Hierarchical BFM")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Nominal coverage", fontsize=12)
    ax.set_ylabel("Empirical coverage", fontsize=12)
    ax.set_title(
        f"Posterior Predictive Calibration\n"
        f"(out-of-fold, N = {n_obs:,})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("POSTERIOR PREDICTIVE CALIBRATION")
    print("=" * 60)

    pred_samples, alea_samples, y_centered = load_inputs()
    n_obs, n_samples = pred_samples.shape
    print(f"OOF observations: {n_obs:,}   posterior samples per obs: {n_samples}")

    # Posterior predictive samples on the centered y-scale (same as
    # pred_samples / y_centered, so coverage is scale-invariant).
    rng = np.random.default_rng(RANDOM_SEED)
    pp_draws = draw_posterior_predictive(pred_samples, alea_samples, rng)

    empirical = coverage_at_levels(pp_draws, y_centered, NOMINAL_LEVELS)

    # Print and save the calibration table
    print(f"\n{'Nominal':>10}  {'Empirical':>10}  {'Gap':>8}")
    print("-" * 32)
    for p, e in zip(NOMINAL_LEVELS, empirical):
        print(f"{p:>10.2f}  {e:>10.3f}  {e - p:>+8.3f}")

    table = pd.DataFrame({
        "nominal_coverage": NOMINAL_LEVELS,
        "empirical_coverage": empirical,
        "gap": empirical - NOMINAL_LEVELS,
    })
    csv_path = OUTPUT_DIR / "calibration_curve.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_calibration(NOMINAL_LEVELS, empirical, n_obs,
                     OUTPUT_DIR / "calibration_curve.png")

    # One-line summary statistic: mean absolute calibration error
    mace = np.mean(np.abs(empirical - NOMINAL_LEVELS))
    print(f"\nMean absolute calibration error: {mace:.4f}")


if __name__ == "__main__":
    main()
