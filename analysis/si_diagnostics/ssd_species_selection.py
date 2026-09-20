"""Separates the BFM-vs-traditional HC20 difference into its components.

Reports, for the Supporting Information:
- the HC20 difference over all species and over the tested species only, which
  splits the total into a species-set effect and a matched-species model effect;
- effective sample sizes for the prediction traces;
- the sampling noise of the replicate-SD statistic.

Reuses the fitted artefacts in outputs/models/; retrains nothing.
Writes ssd_species_selection.csv alongside this script.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data.load_ecotox import load_ecotox_data

DATA_DIR = ROOT / "data" / "raw"
df, y_c, y_mean = load_ecotox_data(
    adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
    chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
    use_molar=False, use_selfies=False, use_mol2vec=False, use_fingerprint=False,
    shuffle=True, random_state=42)

# ---------------- B. names for the 250+ bin -------------------------------
models = ROOT / "outputs" / "models"
df["aleatoric_var"] = np.load(models / "oof_aleatoric.npy")
df["epistemic_var"] = np.load(models / "oof_epistemic.npy")
chem = df.groupby("CAS", observed=True).agg(
    aleatoric_var=("aleatoric_var", "mean"),
    n_obs=("conc", "size"), n_species=("species", "nunique"),
    name=("chem_name", "first")).reset_index()
top = chem[chem["n_obs"] >= 250].sort_values("aleatoric_var", ascending=False)
print("=== B. 250+ bin, top 15 by aleatoric variance ===")
print(top.head(15)[["name", "n_obs", "n_species", "aleatoric_var"]].to_string(index=False))
print(f"\n250+ bin all names:\n{', '.join(top['name'].astype(str).head(50))}")

# ---------------- D. sampling noise of RSD --------------------------------
print("\n=== D. Sampling noise of the RSD / SD estimate ===")
for n in (5, 10, 20, 50, 100):
    # relative SE of the sample SD for normal data ~ 1/sqrt(2(n-1))
    rel_se = 1.0 / np.sqrt(2 * (n - 1))
    # exact: E[s]/sigma = c4
    from scipy.special import gammaln
    c4 = np.exp(0.5 * (np.log(2 / (n - 1)) ) + gammaln(n / 2) - gammaln((n - 1) / 2))
    lo = np.sqrt((n - 1) / chi2.ppf(0.975, n - 1))
    hi = np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
    print(f"  n={n:>4}: E[s]/sigma={c4:.3f} (bias {100*(c4-1):+.1f}%), "
          f"rel.SE(s)={rel_se:.3f}, 95% CI for s/sigma = [{lo:.2f}, {hi:.2f}]")

# observed spread of RSDs in Fig 2 (n>=10 triplets)
g = df.groupby(["CAS", "species", "duration"], observed=True)["conc"]
trip = pd.DataFrame({"n": g.size(), "sd": g.std(ddof=1), "mean": g.mean()}).reset_index()
f2 = trip[trip["n"] >= 10].copy()
f2["rsd"] = (f2["sd"] / f2["mean"]).abs()
print(f"\n  Figure 2 population: {len(f2)} triplets with n>=10, median n={f2['n'].median():.0f}")
print(f"  observed SD (log units) across triplets: median={f2['sd'].median():.3f}, "
      f"IQR=[{f2['sd'].quantile(.25):.3f}, {f2['sd'].quantile(.75):.3f}], "
      f"ratio p75/p25={f2['sd'].quantile(.75)/f2['sd'].quantile(.25):.2f}")
# what spread would pure sampling noise alone produce at the median n?
nmed = int(f2["n"].median())
lo = np.sqrt((nmed - 1) / chi2.ppf(0.75, nmed - 1))
hi = np.sqrt((nmed - 1) / chi2.ppf(0.25, nmed - 1))
print(f"  pure-sampling-noise IQR of s/sigma at n={nmed}: [{lo:.2f}, {hi:.2f}] "
      f"-> ratio {hi/lo:.2f}  (vs observed {f2['sd'].quantile(.75)/f2['sd'].quantile(.25):.2f})")
# Bartlett-style test of homogeneity of variance across triplets
sub = f2.dropna(subset=["sd"])
sub = sub[sub["sd"] > 0]
k = len(sub)
dofs = (sub["n"] - 1).values
s2 = (sub["sd"] ** 2).values
sp2 = np.sum(dofs * s2) / np.sum(dofs)
T = np.sum(dofs) * np.log(sp2) - np.sum(dofs * np.log(s2))
C = 1 + (np.sum(1 / dofs) - 1 / np.sum(dofs)) / (3 * (k - 1))
B = T / C
print(f"  Bartlett test of equal variances across {k} triplets: chi2={B:.0f}, df={k-1}, "
      f"p={chi2.sf(B, k-1):.2e}")

# ---------------- C. ESS of prediction traces ------------------------------
print("\n=== C. ESS of prediction traces (oof_pred_samples) ===")
ps = np.load(models / "oof_pred_samples.npy", mmap_mode="r")
print("  shape:", ps.shape)


def ess1(x):
    x = np.asarray(x, float); n = len(x); x = x - x.mean()
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n].real
    if ac[0] == 0:
        return np.nan, np.nan
    ac /= ac[0]
    s, t = 0.0, 1
    while t + 1 < n:
        p = ac[t] + ac[t + 1]
        if p < 0:
            break
        s += p; t += 2
    tau = 1 + 2 * s
    return n / tau, tau


rng = np.random.default_rng(0)
if ps.ndim == 2:
    nobs = ps.shape[0] if ps.shape[1] > ps.shape[0] else ps.shape[1]
    # figure orientation: samples axis is the one of size 1900
    ax = 0 if ps.shape[0] == 1900 else 1
    idx = rng.choice(ps.shape[1 - ax], size=300, replace=False)
    traces = ps[:, idx].T if ax == 0 else ps[idx, :]
    res = [ess1(t) for t in traces]
    e = np.array([r[0] for r in res]); tau = np.array([r[1] for r in res])
    print(f"  over 300 random held-out predictions: ESS median={np.nanmedian(e):.0f} "
          f"(q10={np.nanpercentile(e,10):.0f}, q90={np.nanpercentile(e,90):.0f}) "
          f"out of {traces.shape[1]} samples; tau_int median={np.nanmedian(tau):.1f}")
else:
    print("  unexpected ndim, raw shape", ps.shape)

# ---------------- A. SSD species-selection test ---------------------------
print("\n=== A. SSD species-selection test (48h) ===")
pred = pd.read_parquet(models / "full_predictions.parquet",
                       columns=["CAS", "species", "duration", "pred_mean"])
pred = pred[pred["duration"] == 48]
print(f"  predictions at 48h: {len(pred)} rows, {pred['CAS'].nunique()} chemicals, "
      f"{pred['species'].nunique()} species")

obs48 = df[df["duration"] == 48]
spp_means = obs48.groupby(["CAS", "species"], observed=True)["conc"].mean().reset_index()
n_spp = spp_means.groupby("CAS", observed=True).size()
elig = n_spp[n_spp >= 5].index
print(f"  chemicals with >=5 tested species at 48h: {len(elig)}")

# traditional lognormal HC20
trad = {}
for cas, gp in spp_means[spp_means["CAS"].isin(elig)].groupby("CAS", observed=True):
    mu, sd = gp["conc"].mean(), gp["conc"].std()
    if sd and not np.isnan(sd) and sd > 0:
        trad[cas] = norm.ppf(0.20, mu, sd)

# BFM HC20 over all species vs over tested species (empirical 20th pct, as in the paper)
pred_e = pred[pred["CAS"].isin(trad.keys())]
hc_all = pred_e.groupby("CAS", observed=True)["pred_mean"].quantile(0.20)
tested = spp_means[["CAS", "species"]].merge(pred_e, on=["CAS", "species"], how="inner")
hc_tested = tested.groupby("CAS", observed=True)["pred_mean"].quantile(0.20)

out = pd.DataFrame({"trad": pd.Series(trad), "bfm_all": hc_all,
                    "bfm_tested": hc_tested}).dropna()
out["n_species"] = n_spp.reindex(out.index)
out["d_all"] = out["bfm_all"] - out["trad"]
out["d_tested"] = out["bfm_tested"] - out["trad"]
out["d_species_selection"] = out["bfm_all"] - out["bfm_tested"]
print(f"\n  N chemicals compared: {len(out)}")
print(f"  median(BFM_all - trad)              = {out['d_all'].median():+.3f} log mg/L   "
      f"(mean {out['d_all'].mean():+.3f})")
print(f"  median(BFM_tested - trad)           = {out['d_tested'].median():+.3f} log mg/L   "
      f"(mean {out['d_tested'].mean():+.3f})")
print(f"  median(BFM_all - BFM_tested)        = {out['d_species_selection'].median():+.3f} log mg/L "
      f"<- pure species-set effect")
print(f"\n  fraction with BFM_all < trad:    {(out['d_all']<0).mean():.1%}")
print(f"  fraction with BFM_tested < trad: {(out['d_tested']<0).mean():.1%}")
print(f"\n  spearman(n_species, d_all)    = "
      f"{out[['n_species','d_all']].corr(method='spearman').iloc[0,1]:+.3f}")
print(f"  spearman(n_species, d_tested) = "
      f"{out[['n_species','d_tested']].corr(method='spearman').iloc[0,1]:+.3f}")

# also: traditional lognormal vs empirical 20th percentile of the SAME observed species
emp_trad = spp_means[spp_means["CAS"].isin(out.index)].groupby("CAS", observed=True)["conc"].quantile(0.20)
out["trad_empirical"] = emp_trad
out["d_lognormal_assumption"] = out["trad"] - out["trad_empirical"]
print(f"\n  median(lognormal HC20 - empirical 20th pct of same observed species) = "
      f"{out['d_lognormal_assumption'].median():+.3f}  <- pure estimator/parametric effect")
out.to_csv(OUT_DIR / "ssd_species_selection.csv")
print(f"\n  saved {OUT_DIR / 'ssd_species_selection.csv'}")
