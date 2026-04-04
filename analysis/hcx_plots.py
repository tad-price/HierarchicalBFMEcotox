"""
hcx_plots.py - HC20 Comparison Plots (Figures 10 & 11)

Reads the HCx comparison CSV produced by ssd_mc_uncertainty.py --all-hcx
and generates:
- Figure 10: Forest plot of BFM HC20 with 95% CI vs traditional HC20
- Figure 11: Correlation scatter of traditional vs BFM HC20

Requires:
- outputs/figures/ssd_analysis/hcx_comparison_48h.csv

Outputs to:
- outputs/figures/ssd_analysis/

Usage:
    python analysis/hcx_plots.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures" / "ssd_analysis"

DURATION_HOURS = 48
PERCENTILE = 20  # HC20


def load_comparison_data():
    """Load the merged HCx comparison CSV."""
    path = OUTPUT_DIR / f"hcx_comparison_{DURATION_HOURS}h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"HCx comparison CSV not found at {path}\n"
            f"Run: python analysis/ssd_mc_uncertainty.py --all-hcx --percentiles {PERCENTILE}"
        )
    df = pd.read_csv(path)
    return df


# =============================================================================
# FIGURE 10: Forest plot
# =============================================================================

def figure10_forest_plot(df, n_display=30):
    """
    Forest plot of BFM-derived HC20 with 95% CI vs traditional HC20.

    Shows a subset of chemicals spanning the HC20 range, sorted by BFM HC20.
    """
    tag = f"hc{PERCENTILE}"
    mc_col = f"{tag}_mean"
    ci_lo = f"{tag}_ci_lower"
    ci_hi = f"{tag}_ci_upper"
    trad_col = f"trad_{tag}"

    # Keep only chemicals that have both BFM and traditional values
    plot_df = df.dropna(subset=[mc_col, trad_col]).copy()
    plot_df = plot_df.sort_values(mc_col).reset_index(drop=True)

    # Subsample evenly across the range to avoid overcrowding
    if len(plot_df) > n_display:
        indices = np.linspace(0, len(plot_df) - 1, n_display, dtype=int)
        plot_df = plot_df.iloc[indices].reset_index(drop=True)

    # Truncate long names
    plot_df["label"] = plot_df["chemical_name"].fillna(plot_df["CAS"]).str[:25]

    fig, ax = plt.subplots(figsize=(10, 10))

    y_pos = np.arange(len(plot_df))

    # CI bars
    xerr_lo = plot_df[mc_col].values - plot_df[ci_lo].values
    xerr_hi = plot_df[ci_hi].values - plot_df[mc_col].values
    ax.errorbar(
        plot_df[mc_col], y_pos, xerr=[xerr_lo, xerr_hi],
        fmt='o', color='steelblue', markersize=5, elinewidth=1.5,
        capsize=0, alpha=0.8, label="MC HC20 (mean)",
    )

    # CI shading
    for i, row in plot_df.iterrows():
        idx = plot_df.index.get_loc(i)
        ax.fill_betweenx(
            [idx - 0.3, idx + 0.3], row[ci_lo], row[ci_hi],
            color='steelblue', alpha=0.15,
        )

    # Traditional HC20
    ax.scatter(
        plot_df[trad_col], y_pos, marker='D', color='firebrick',
        s=40, zorder=5, label="Traditional HC20",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel(f"HC{PERCENTILE} (log mg/L)", fontsize=12)
    ax.set_title(
        f"MC Posterior HC{PERCENTILE} with 95% CI vs Traditional HC{PERCENTILE}\n"
        f"({len(plot_df)} chemicals spanning the HC{PERCENTILE} range)",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"hc{PERCENTILE}_forest_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 11: Correlation scatter
# =============================================================================

def figure11_correlation_scatter(df):
    """
    Scatter plot of traditional vs BFM HC20, coloured by number of observed species.
    """
    tag = f"hc{PERCENTILE}"
    mc_col = f"{tag}_mean"
    trad_col = f"trad_{tag}"

    plot_df = df.dropna(subset=[mc_col, trad_col]).copy()
    corr = plot_df[mc_col].corr(plot_df[trad_col])

    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        plot_df[trad_col], plot_df[mc_col],
        c=plot_df["n_species_observed"], cmap="viridis",
        s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
        norm=plt.matplotlib.colors.LogNorm(),
    )
    cbar = plt.colorbar(sc, ax=ax, label="Observed species (traditional SSD)")

    # 1:1 line
    lims = [
        min(plot_df[trad_col].min(), plot_df[mc_col].min()) - 0.5,
        max(plot_df[trad_col].max(), plot_df[mc_col].max()) + 0.5,
    ]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.6, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"Traditional SSD HC{PERCENTILE} (log mg/L)", fontsize=12)
    ax.set_ylabel(f"MC Posterior HC{PERCENTILE} (log mg/L)", fontsize=12)
    ax.set_title(
        f"HC{PERCENTILE} Comparison: MC Posterior vs Traditional SSD\n"
        f"n = {len(plot_df)} chemicals, r = {corr:.3f}",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Annotation box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.97, 0.03, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=11,
            ha='right', va='bottom', bbox=props)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"hc{PERCENTILE}_correlation_trad_vs_mc.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print(f"  Pearson r = {corr:.3f}")
    print(f"  N chemicals = {len(plot_df)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"HC{PERCENTILE} COMPARISON PLOTS")
    print("=" * 60)

    df = load_comparison_data()
    figure10_forest_plot(df)
    figure11_correlation_scatter(df)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
