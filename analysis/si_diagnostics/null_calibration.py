"""Simulates the predicted/empirical SD ratio under perfect calibration.

The empirical SD of a triplet with few replicates is noisy and biased low, so
the ratio in the replicate-calibration table exceeds one even for a
well-calibrated model. This script simulates that null ratio at the observed
replicate counts, giving the Null Median column of that table.

Reuses the fitted artefacts in outputs/models/; retrains nothing.
Writes table2_null_calibration.csv alongside this script.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.load_ecotox import load_ecotox_data

DATA_DIR = ROOT / "data" / "raw"
df, y_c, y_mean = load_ecotox_data(
    adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
    chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
    use_molar=False, use_selfies=False, use_mol2vec=False, use_fingerprint=False,
    shuffle=True, random_state=42)
models = ROOT / "outputs" / "models"
df["aleatoric_var"] = np.load(models / "oof_aleatoric.npy")

# ---------- E. null-calibration simulation ---------------------------------
g = df.groupby(["CAS", "species", "duration"], observed=True)
trip = pd.DataFrame({
    "n": g["conc"].size(),
    "emp_sd": g["conc"].std(ddof=1),
    "pred_sd": g["aleatoric_var"].mean().pipe(np.sqrt),
}).reset_index()
trip = trip[(trip["n"] >= 5) & trip["emp_sd"].notna() & (trip["emp_sd"] > 0)]
print(f"E. triplets with >=5 replicates: {len(trip)}")

BINS = [(5, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]
LAB = ["5-9", "10-19", "20-49", "50-99", "100+"]
rng = np.random.default_rng(0)
NSIM = 200

print(f"\n{'bin':>7} {'N':>6} | {'OBSERVED':>19} | {'NULL (perfect calib.)':>25} | {'c4-corrected obs':>17}")
print(f"{'':>7} {'':>6} | {'medRatio':>9} {'pooled':>9} | {'medRatio':>9} {'IQR':>15} | {'medRatio':>17}")
rows = []
for (lo, hi), lab in zip(BINS, LAB):
    s = trip[(trip["n"] >= lo) & (trip["n"] <= hi)]
    if not len(s):
        continue
    obs_med = (s["pred_sd"] / s["emp_sd"]).median()
    pooled = s["pred_sd"].mean() / s["emp_sd"].mean()
    n_arr = s["n"].values
    sd_arr = s["pred_sd"].values
    # c4 bias correction of the empirical SD
    c4 = np.exp(0.5 * np.log(2 / (n_arr - 1)) + gammaln(n_arr / 2) - gammaln((n_arr - 1) / 2))
    obs_med_c4 = np.median(s["pred_sd"].values / (s["emp_sd"].values / c4))
    # simulate: truth = pred_sd, draw n replicates, recompute ratio
    sims = []
    for _ in range(NSIM):
        s_hat = np.array([sd * rng.standard_normal(nn).std(ddof=1)
                          for sd, nn in zip(sd_arr, n_arr)])
        sims.append(np.median(sd_arr / s_hat))
    sims = np.array(sims)
    print(f"{lab:>7} {len(s):>6} | {obs_med:>9.3f} {pooled:>9.3f} | {sims.mean():>9.3f} "
          f"[{np.percentile(sims,2.5):.3f},{np.percentile(sims,97.5):.3f}] | {obs_med_c4:>17.3f}")
    rows.append({"bin": lab, "N": len(s), "obs_median_ratio": round(obs_med, 3),
                 "obs_pooled_ratio": round(pooled, 3),
                 "null_median_ratio": round(sims.mean(), 3),
                 "null_ci": f"[{np.percentile(sims,2.5):.3f}, {np.percentile(sims,97.5):.3f}]",
                 "obs_median_ratio_c4corrected": round(obs_med_c4, 3)})
pd.DataFrame(rows).to_csv(Path(__file__).parent / "table2_null_calibration.csv", index=False)

# ---------- F. verify HC20 claims from the paper's own CSV -----------------
print("\n=== F. HC20 claims vs the paper's own CSV ===")
h = pd.read_csv(ROOT / "outputs/figures/hcx_comparison_48h.csv").dropna(subset=["trad_hc20"])
h["d"] = h["hc20_mean"] - h["trad_hc20"]
print(f"  N={len(h)}; pearson r(trad, hc20_mean) = "
      f"{h[['trad_hc20','hc20_mean']].corr().iloc[0,1]:.3f}")
print(f"  median(BFM - trad) = {h['d'].median():+.3f}, mean = {h['d'].mean():+.3f} log mg/L")
print(f"  fraction below 1:1 line = {(h['d']<0).mean():.1%}")
print(f"  spearman(n_species_observed, d) = "
      f"{h[['n_species_observed','d']].corr(method='spearman').iloc[0,1]:+.3f}")
print(f"  fraction where trad falls OUTSIDE the BFM 95% CI = "
      f"{((h['trad_hc20']<h['hc20_ci_lower'])|(h['trad_hc20']>h['hc20_ci_upper'])).mean():.1%}")

# ---------- G. RSD on log-transformed data ---------------------------------
print("\n=== G. RSD definition on log-transformed data (Figure 2) ===")
t2 = pd.DataFrame({"n": g["conc"].size(), "sd": g["conc"].std(ddof=1),
                   "mean": g["conc"].mean()}).reset_index()
t2 = t2[t2["n"] >= 10].dropna()
t2["rsd"] = (t2["sd"] / t2["mean"]).abs()
near0 = (t2["mean"].abs() < 0.5).mean()
print(f"  {len(t2)} triplets; |mean log LC50| < 0.5 for {near0:.1%} of them "
      f"(LC50 between 0.32 and 3.2 mg/L) -> RSD denominator near zero")
print(f"  RSD: median={t2['rsd'].median():.2f}, p90={t2['rsd'].quantile(.90):.2f}, "
      f"max={t2['rsd'].max():.1f}")
print(f"  RSD>1 for {(t2['rsd']>1).mean():.1%} of triplets; "
      f"of those, {(t2.loc[t2['rsd']>1,'mean'].abs()<0.5).mean():.1%} have |mean log|<0.5")
print(f"  by contrast, raw SD of log LC50: median={t2['sd'].median():.3f}, "
      f"p90={t2['sd'].quantile(.90):.3f}, max={t2['sd'].max():.2f}")
print(f"  spearman(|mean|, RSD) = {t2.assign(am=t2['mean'].abs())[['am','rsd']].corr(method='spearman').iloc[0,1]:+.3f}  "
      f"(strong negative => RSD driven by the denominator, not by real noise)")
