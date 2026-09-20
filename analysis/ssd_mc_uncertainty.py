"""
ssd_mc_uncertainty.py - SSD Uncertainty via MCMC Posterior Samples

This script generates Species Sensitivity Distributions with uncertainty bands
by using the full MCMC posterior samples from the trained Hierarchical BFM.

Unlike independent Monte Carlo sampling from summary statistics (mean, variance),
this approach preserves the correlation structure between species predictions:
each posterior sample produces a complete, internally consistent set of predictions
for all species, so the resulting SSD curves properly capture epistemic uncertainty.

Requires:
- outputs/models/trained_model.pkl
- outputs/models/full_predictions.parquet

Writes to outputs/figures/.

Usage:
    python analysis/ssd_mc_uncertainty.py --cas 1912-24-9
    python analysis/ssd_mc_uncertainty.py --cas 14437-17-3
    python analysis/ssd_mc_uncertainty.py --all-hcx --percentiles 20
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import scipy.sparse as sp
from scipy.stats import norm

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Shared utilities from ssd_analysis; the target chemical is set via set_target().
from ssd_analysis import (
    DURATION_HOURS, OUTPUT_DIR,
    load_observations, load_full_predictions,
    filter_observations, filter_predictions,
    set_target,
)
import ssd_analysis


# =============================================================================
# SSD WITH MCMC POSTERIOR SAMPLES
# =============================================================================

def plot_ssd_with_uncertainty(pred_df, df_obs, n_curves=2000, include_aleatoric=False, seed=42):
    """
    Generate an SSD with uncertainty by plotting individual MCMC posterior sample SSDs.
    
    Methodology:
    - Load the trained model with all posterior samples
    - Build a design matrix for the target chemical at the target duration
    - For each posterior sample:
        1. Compute predictions for all species using that sample's parameters
           (this preserves correlated epistemic structure — all species shift together)
        2. Optionally add independent aleatoric noise from that sample's alpha_c
        3. Sort predictions to get an SSD curve
        4. Plot with high transparency
    
    This properly captures both:
    - Epistemic uncertainty (correlated shifts across species from shared parameters)
    - Aleatoric uncertainty (independent noise per species from learned alpha_c)
    
    The posterior ensemble is summarised as a 95% credible band and a median SSD,
    and the traditional lognormal SSD (fitted to the observed species) is overlaid
    on the same axes for direct comparison.

    Args:
        pred_df: Predictions for target chemical (used for the species list and metadata)
        df_obs: Observations for target chemical (for the traditional lognormal overlay)
        n_curves: Number of posterior samples to use (default 2000, subsampled if needed)
        include_aleatoric: Whether to add aleatoric noise draws (default False)
        seed: Random seed for reproducibility
    """
    print("\n" + "="*60)
    print(f"SSD WITH MCMC POSTERIOR SAMPLES ({n_curves} curves)")
    print("="*60)
    
    rng = np.random.default_rng(seed)
    
    # Load the trained model
    model_path = ROOT_DIR / "outputs" / "models" / "trained_model.pkl"
    print(f"Loading trained model from {model_path}...")
    model_data = joblib.load(model_path)
    
    model = model_data["model"]
    enc_dict = model_data["enc_dict"]
    imputer = model_data["imputer"]
    scaler = model_data["scaler"]
    cas_to_idx = model_data["cas_to_idx"]
    species_to_tax = model_data["species_to_tax"]
    chem_props = model_data["chem_props"]
    num_cols = model_data["num_cols"]
    unique_species = model_data["unique_species"]
    y_mean = model_data.get("y_mean", 0.0)

    n_total_samples = len(model.samples)
    print(f"Model has {n_total_samples} posterior samples")
    
    # Build design matrix for target chemical at target duration, all species
    rows = []
    for species in unique_species:
        rows.append({
            "CAS": ssd_analysis.TARGET_CAS,
            "species": species,
            "duration": DURATION_HOURS
        })
    
    grid_df = pd.DataFrame(rows)
    grid_df["duration"] = pd.Categorical(grid_df["duration"].astype(int), 
                                          categories=model_data["unique_durations"])
    
    # Add taxonomy and chemical properties
    grid_df = grid_df.merge(species_to_tax, on="species", how="left")
    grid_df["tax_family"] = grid_df["tax_family"].fillna("Unknown")
    grid_df["tax_class"] = grid_df["tax_class"].fillna("Unknown")
    grid_df = grid_df.merge(chem_props, on="CAS", how="left")
    
    # Build design matrix
    def make_design_cats(df, enc):
        Xi = enc["species"].transform(df[["species"]])
        Xj = enc["CAS"].transform(df[["CAS"]])
        Xd = enc["duration"].transform(df[["duration"]])
        Xt = enc["tax_family"].transform(df[["tax_family"]])
        Xe = enc["tax_class"].transform(df[["tax_class"]])
        return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")
    
    X_cat = make_design_cats(grid_df, enc_dict)
    num_grid = scaler.transform(imputer.transform(grid_df[num_cols]))
    X_grid = sp.hstack([X_cat, sp.csr_matrix(num_grid)], format="csr")
    X2_grid = X_grid.copy(); X2_grid.data **= 2
    
    chem_group_idx = cas_to_idx[ssd_analysis.TARGET_CAS]
    n_species = len(unique_species)
    
    print(f"Built design matrix: {X_grid.shape}")
    print(f"Chemical group index: {chem_group_idx}")
    
    # Subsample posterior samples if needed
    if n_curves < n_total_samples:
        sample_indices = rng.choice(n_total_samples, size=n_curves, replace=False)
    else:
        sample_indices = np.arange(n_total_samples)
        n_curves = n_total_samples
    
    # Compute predictions for each posterior sample
    print(f"Computing predictions for {n_curves} posterior samples...")
    all_ssd_curves = np.zeros((n_curves, n_species))
    
    for i, idx in enumerate(sample_indices):
        s = model.samples[idx]
        w0, w, v, alpha_vec = s["w0"], s["w"], s["v"], s["alpha_vec"]
        
        # Deterministic prediction from this sample's parameters
        q = X_grid @ v
        inter = 0.5 * ((q**2) - X2_grid @ (v**2)).sum(axis=1)
        preds = np.asarray(w0 + X_grid @ w + inter).ravel() + y_mean

        # Optionally add aleatoric noise using this sample's alpha_c
        if include_aleatoric:
            aleatoric_sd = np.sqrt(1.0 / alpha_vec[chem_group_idx])
            preds = preds + rng.normal(0, aleatoric_sd, size=n_species)

        # Sort to get SSD curve
        all_ssd_curves[i] = np.sort(preds)

    # Empirical CDF y-values
    y_cdf = np.arange(1, n_species + 1) / (n_species + 1)
    
    # Ensemble summary statistics
    median_curve = np.percentile(all_ssd_curves, 50, axis=0)
    lower_95 = np.percentile(all_ssd_curves, 2.5, axis=0)
    upper_95 = np.percentile(all_ssd_curves, 97.5, axis=0)

    # Traditional lognormal SSD fitted to observed species means (for overlay).
    # Uses the same mean/std (ddof=1) definition as compute_traditional_hcx so the
    # overlay curve and the aggregate HC comparison are numerically consistent.
    tox_obs = df_obs.groupby("species", observed=True)["y_true"].mean().dropna().values
    mu_trad = tox_obs.mean()
    std_trad = tox_obs.std(ddof=1)
    n_obs = len(tox_obs)
    y_obs = np.arange(1, n_obs + 1) / (n_obs + 1)

    # Data-driven x-limits spanning the ensemble band and the observations
    x_lo = float(min(lower_95.min(), tox_obs.min())) - 0.5
    x_hi = float(max(upper_95.max(), tox_obs.max())) + 0.5
    x_range = np.linspace(x_lo, x_hi, 300)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ensemble: shaded 95% credible band + median SSD
    ax.fill_betweenx(y_cdf, lower_95, upper_95, color='steelblue', alpha=0.25,
                     label='BFM 95% credible band')
    ax.plot(median_curve, y_cdf, color='navy', linewidth=2, zorder=4,
            label='BFM ensemble median')

    # Traditional lognormal SSD + observed species
    ax.plot(x_range, norm.cdf(x_range, mu_trad, std_trad), 'r-', linewidth=2, zorder=3,
            label=f'Traditional SSD (n={n_obs})')
    ax.scatter(np.sort(tox_obs), y_obs, c='black', s=40, zorder=5,
               edgecolors='white', linewidths=0.5, label='Observed species')

    noise_label = "epistemic + aleatoric" if include_aleatoric else "epistemic only"
    ax.set_xlabel("Predicted Toxicity (Log mg/L)", fontsize=12)
    ax.set_ylabel("Fraction of Species Affected", fontsize=12)
    ax.set_title(f"SSD with Posterior Uncertainty: {ssd_analysis.CHEMICAL_NAME} at {DURATION_HOURS}h\n"
                 f"({n_curves} MCMC samples, {noise_label})", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(x_lo, x_hi)
    
    safe_name = ssd_analysis.CHEMICAL_NAME.lower().replace(' ', '_').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    output_path = OUTPUT_DIR / f"ssd_uncertainty_{safe_name}_{DURATION_HOURS}h.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # HC5 results. Concentrations are log10 mg/L, so mg/L is 10**x, not exp(x).
    print(f"\n{'='*60}")
    print("HC5 RESULTS (5% of species affected)")
    print(f"{'='*60}")
    
    idx_hc5 = int(0.05 * n_species)
    hc5_median = median_curve[idx_hc5]
    hc5_lower = lower_95[idx_hc5]
    hc5_upper = upper_95[idx_hc5]
    hc5_width = hc5_upper - hc5_lower
    
    print(f"\n{'Metric':<20} {'Log mg/L':<15} {'mg/L':<15}")
    print(f"{'-'*50}")
    print(f"{'HC5 Median':<20} {hc5_median:<15.3f} {10.0 ** hc5_median:<15.6f}")
    print(f"{'HC5 Lower (2.5%)':<20} {hc5_lower:<15.3f} {10.0 ** hc5_lower:<15.6f}")
    print(f"{'HC5 Upper (97.5%)':<20} {hc5_upper:<15.3f} {10.0 ** hc5_upper:<15.6f}")
    print(f"{'-'*50}")
    print(f"{'95% CI Width':<20} {hc5_width:<15.3f}")

    # HC20 (the metric reported in the paper's hazard-concentration section)
    idx_hc20 = int(0.20 * n_species)
    trad_hc20 = norm.ppf(0.20, loc=mu_trad, scale=std_trad)
    print(f"\n{'HC20 Median (BFM)':<22} {median_curve[idx_hc20]:<15.3f}")
    print(f"{'HC20 95% CI (BFM)':<22} [{lower_95[idx_hc20]:.3f}, {upper_95[idx_hc20]:.3f}]")
    print(f"{'HC20 Traditional':<22} {trad_hc20:<15.3f}")

    return hc5_median, hc5_lower, hc5_upper


# =============================================================================
# HCx FOR ALL CHEMICALS
# =============================================================================

def compute_hcx_all_chemicals(percentiles=[20], n_samples=None,
                               include_aleatoric=False, seed=42):
    """
    Compute HCx values for ALL chemicals using MCMC posterior samples.

    For each chemical and each posterior sample:
      1. Build a design matrix for all species at the target duration
      2. Predict toxicity for all species using that sample's parameters
      3. Extract HCx values (e.g. 5th and 20th percentile of predictions)

    Across the N posterior samples this yields N HCx values per chemical,
    summarised as mean, std, min, max, median, and 95% CI.

    Args:
        percentiles: List of HCx percentiles to compute (default [20])
        n_samples:   Number of posterior samples to use (None = all)
        include_aleatoric: Whether to add aleatoric noise draws (default False)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with one row per chemical and summary columns per HCx.
    """
    from tqdm.auto import tqdm

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = ROOT_DIR / "outputs" / "models" / "trained_model.pkl"
    print(f"Loading trained model from {model_path}...")
    model_data = joblib.load(model_path)

    model          = model_data["model"]
    enc_dict       = model_data["enc_dict"]
    imputer        = model_data["imputer"]
    scaler         = model_data["scaler"]
    cas_to_idx     = model_data["cas_to_idx"]
    species_to_tax = model_data["species_to_tax"]
    chem_props     = model_data["chem_props"]
    num_cols       = model_data["num_cols"]
    unique_species = model_data["unique_species"]
    y_mean         = model_data.get("y_mean", 0.0)

    n_total_samples = len(model.samples)
    print(f"Model has {n_total_samples} posterior samples")

    # Decide how many samples to use
    if n_samples is None or n_samples >= n_total_samples:
        sample_indices = np.arange(n_total_samples)
        n_samples = n_total_samples
    else:
        sample_indices = rng.choice(n_total_samples, size=n_samples, replace=False)

    # ------------------------------------------------------------------
    # CAS → chemical name mapping (from the observations file)
    # ------------------------------------------------------------------
    try:
        from data.load_ecotox import load_ecotox_data
        DATA_DIR = ROOT_DIR / "data" / "raw"
        obs_df, _, _ = load_ecotox_data(
            adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
            chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
            use_molar=False, use_selfies=False,
            use_mol2vec=False, use_fingerprint=False,
            shuffle=False, random_state=42,
        )
        cas_to_name = (
            obs_df[["CAS", "chem_name"]]
            .drop_duplicates("CAS")
            .set_index("CAS")["chem_name"]
            .to_dict()
        )
    except Exception:
        cas_to_name = {}

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------
    all_cas = sorted(cas_to_idx.keys())
    n_species = len(unique_species)
    n_chemicals = len(all_cas)

    print(f"\nComputing HCx for {n_chemicals} chemicals × {n_samples} posterior samples")
    print(f"  Species per SSD: {n_species}")
    print(f"  Duration: {DURATION_HOURS}h")
    print(f"  Percentiles: {percentiles}")
    print(f"  Aleatoric noise: {'yes' if include_aleatoric else 'no'}")

    # Design-matrix helper (same structure as in plot_ssd_with_uncertainty)
    def make_design_cats(df, enc):
        Xi = enc["species"].transform(df[["species"]])
        Xj = enc["CAS"].transform(df[["CAS"]])
        Xd = enc["duration"].transform(df[["duration"]])
        Xt = enc["tax_family"].transform(df[["tax_family"]])
        Xe = enc["tax_class"].transform(df[["tax_class"]])
        return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

    # Pre-extract sample parameter arrays for faster inner loop
    sample_list = [model.samples[i] for i in sample_indices]

    # ------------------------------------------------------------------
    # Main loop over chemicals
    # ------------------------------------------------------------------
    results = []

    for cas in tqdm(all_cas, desc="HCx all chemicals"):
        # --- build design matrix for this chemical ---
        grid_df = pd.DataFrame({
            "CAS": cas,
            "species": unique_species,
            "duration": DURATION_HOURS,
        })
        grid_df["duration"] = pd.Categorical(
            grid_df["duration"].astype(int),
            categories=model_data["unique_durations"],
        )
        grid_df = grid_df.merge(species_to_tax, on="species", how="left")
        grid_df["tax_family"] = grid_df["tax_family"].fillna("Unknown")
        grid_df["tax_class"]  = grid_df["tax_class"].fillna("Unknown")
        grid_df = grid_df.merge(chem_props, on="CAS", how="left")

        X_cat   = make_design_cats(grid_df, enc_dict)
        num_raw = scaler.transform(imputer.transform(grid_df[num_cols]))
        X_grid  = sp.hstack([X_cat, sp.csr_matrix(num_raw)], format="csr")
        X2_grid = X_grid.copy(); X2_grid.data **= 2

        chem_group_idx = cas_to_idx[cas]

        # --- compute HCx per posterior sample ---
        hcx_samples = {p: np.empty(n_samples) for p in percentiles}

        for i, s in enumerate(sample_list):
            w0, w, v, alpha_vec = s["w0"], s["w"], s["v"], s["alpha_vec"]

            q = X_grid @ v
            inter = 0.5 * ((q ** 2) - X2_grid @ (v ** 2)).sum(axis=1)
            preds = np.asarray(w0 + X_grid @ w + inter).ravel() + y_mean

            if include_aleatoric:
                aleatoric_sd = np.sqrt(1.0 / alpha_vec[chem_group_idx])
                preds = preds + rng.normal(0, aleatoric_sd, size=n_species)

            for p in percentiles:
                hcx_samples[p][i] = np.percentile(preds, p)

        # --- summarise ---
        row = {
            "CAS": cas,
            "chemical_name": cas_to_name.get(cas, ""),
        }
        for p in percentiles:
            vals = hcx_samples[p]
            tag = f"hc{p}"
            row[f"{tag}_mean"]     = vals.mean()
            row[f"{tag}_median"]   = np.median(vals)
            row[f"{tag}_std"]      = vals.std()
            row[f"{tag}_min"]      = vals.min()
            row[f"{tag}_max"]      = vals.max()
            row[f"{tag}_ci_lower"] = np.percentile(vals, 2.5)
            row[f"{tag}_ci_upper"] = np.percentile(vals, 97.5)

        results.append(row)

    # ------------------------------------------------------------------
    # Build output DataFrame and save
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    aleatoric_tag = "_with_aleatoric" if include_aleatoric else ""
    output_path = OUTPUT_DIR / f"hcx_all_chemicals_{DURATION_HOURS}h{aleatoric_tag}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"Shape: {df_results.shape}")

    # Quick summary
    for p in percentiles:
        tag = f"hc{p}"
        ci_widths = df_results[f"{tag}_ci_upper"] - df_results[f"{tag}_ci_lower"]
        print(f"\n  HC{p} summary (log mg/L):")
        print(f"    Grand mean of means : {df_results[f'{tag}_mean'].mean():.3f}")
        print(f"    Mean posterior SD    : {df_results[f'{tag}_std'].mean():.3f}")
        print(f"    Mean 95% CI width   : {ci_widths.mean():.3f}")

    return df_results


# =============================================================================
# TRADITIONAL SSD HCx (normal fit to observed species means)
# =============================================================================

def compute_traditional_hcx(df_obs, percentiles=[20], min_species=5):
    """
    Compute traditional HCx for all chemicals with enough observations.

    Methodology (per chemical at DURATION_HOURS):
    - Aggregate observations to one value per species (mean of log mg/L)
    - Require at least `min_species` species
    - Fit a normal distribution to those species means
    - HCx = norm.ppf(percentile/100, mu, std)

    Args:
        df_obs: Full observations DataFrame (must have CAS, species, duration, y_true)
        percentiles: List of HCx percentiles (default [20])
        min_species: Minimum number of species required (default 5)

    Returns:
        DataFrame with traditional HCx values (one row per qualifying chemical)
    """
    # Filter to target duration
    df = df_obs[df_obs["duration"] == DURATION_HOURS].copy()

    results = []
    for cas, group in df.groupby("CAS", observed=True):
        # One toxicity value per species (mean of observations)
        species_means = group.groupby("species", observed=True)["y_true"].mean()
        n_spp = len(species_means)

        if n_spp < min_species:
            continue

        mu  = species_means.mean()
        std = species_means.std()

        if std == 0 or np.isnan(std):
            continue

        row = {"CAS": cas, "n_species_observed": n_spp}
        for p in percentiles:
            row[f"trad_hc{p}"] = norm.ppf(p / 100.0, loc=mu, scale=std)

        results.append(row)

    df_trad = pd.DataFrame(results)
    print(f"Traditional HCx computed for {len(df_trad)} chemicals "
          f"(≥{min_species} species at {DURATION_HOURS}h)")
    return df_trad


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SSD Monte Carlo Uncertainty Analysis")
    parser.add_argument("--cas", type=str, default="1912-24-9",
                        help="CAS number of target chemical (default: 1912-24-9 = Atrazine)")
    parser.add_argument("--name", type=str, default=None,
                        help="Chemical name (auto-detected from data if not provided)")
    parser.add_argument("--all-hcx", action="store_true",
                        help="Compute HC20 for ALL chemicals (instead of plotting one SSD)")
    parser.add_argument("--percentiles", type=int, nargs="+", default=[20],
                        help="HCx percentiles to compute (default: 20)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of posterior samples to use (default: 200)")
    parser.add_argument("--include-aleatoric", action="store_true",
                        help="Add aleatoric noise draws to predictions")
    parser.add_argument("--min-species", type=int, default=5,
                        help="Min species for traditional SSD (default: 5)")
    args = parser.parse_args()

    set_target(args.cas, args.name)

    if args.all_hcx:
        # ---- All-chemicals HCx mode ----
        print("="*60)
        print("HCx COMPUTATION FOR ALL CHEMICALS")
        print(f"Duration: {DURATION_HOURS}h")
        print("="*60)

        # 1. MC posterior HC20 for all chemicals
        df_mc = compute_hcx_all_chemicals(
            percentiles=args.percentiles,
            n_samples=args.n_samples,
            include_aleatoric=args.include_aleatoric,
        )

        # 2. Traditional HC20 for chemicals with enough observations
        print("\nComputing traditional HCx...")
        df_obs = load_observations()
        df_trad = compute_traditional_hcx(
            df_obs,
            percentiles=args.percentiles,
            min_species=args.min_species,
        )

        # 3. Merge on CAS
        df_merged = df_mc.merge(df_trad, on="CAS", how="left")

        aleatoric_tag = "_with_aleatoric" if args.include_aleatoric else ""
        merged_path = OUTPUT_DIR / f"hcx_comparison_{DURATION_HOURS}h{aleatoric_tag}.csv"
        df_merged.to_csv(merged_path, index=False)
        print(f"\nSaved merged comparison to {merged_path}")
        print(f"  Total chemicals:               {len(df_merged)}")
        print(f"  With traditional SSD available: {df_merged['n_species_observed'].notna().sum()}")

        # 4. Quick comparison for overlapping chemicals
        for p in args.percentiles:
            mc_col   = f"hc{p}_mean"
            trad_col = f"trad_hc{p}"
            overlap = df_merged.dropna(subset=[trad_col])
            if len(overlap) == 0:
                continue
            diff = overlap[mc_col] - overlap[trad_col]
            corr = overlap[mc_col].corr(overlap[trad_col])
            print(f"\n  HC{p} comparison ({len(overlap)} chemicals):")
            print(f"    Pearson r:     {corr:.3f}")
            print(f"    Mean diff (MC − trad): {diff.mean():.3f}")
            print(f"    Std diff:      {diff.std():.3f}")

    else:
        # ---- Single-chemical SSD plot mode ----
        print("="*60)
        print("SSD MONTE CARLO UNCERTAINTY ANALYSIS")
        print(f"Target: CAS {ssd_analysis.TARGET_CAS} at {DURATION_HOURS}h")
        print("="*60)

        # Load data (filter_observations will auto-detect CHEMICAL_NAME)
        df_obs = load_observations()
        pred_df = load_full_predictions()

        # Filter to target chemical
        df_obs_filtered = filter_observations(df_obs)
        pred_df_filtered = filter_predictions(pred_df)

        if len(pred_df_filtered) == 0:
            print("ERROR: No predictions for the specified chemical and duration!")
            sys.exit(1)

        # Generate MCMC uncertainty SSD
        hc5_model, hc5_lower, hc5_upper = plot_ssd_with_uncertainty(pred_df_filtered, df_obs_filtered)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Chemical: {ssd_analysis.CHEMICAL_NAME}")
        print(f"Duration: {DURATION_HOURS}h")
        print(f"Predicted species: {len(pred_df_filtered)}")
        print("\nMCMC HC5:")
        print(f"  Median:  {hc5_model:.3f} ({10.0 ** hc5_model:.6f} mg/L)")
        print(f"  95% CI:  [{hc5_lower:.3f}, {hc5_upper:.3f}]")
        print(f"  CI Width: {hc5_upper - hc5_lower:.3f}")

        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()
