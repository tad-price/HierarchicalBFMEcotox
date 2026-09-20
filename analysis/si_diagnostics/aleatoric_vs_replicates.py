"""Compares the learned per-chemical noise term with genuine replicate scatter.

Reports, for the Supporting Information:
- aleatoric variance against the pooled within-triplet replicate variance, over
  the chemicals with at least ten replicate degrees of freedom;
- the composition of the highest observation-count bin;
- the autocorrelation and effective sample size of the b0 chain.

Reuses the fitted artefacts in outputs/models/; retrains nothing.
Writes ale_vs_within.csv alongside this script.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
df["epistemic_var"] = np.load(models / "oof_epistemic.npy")
df["oof_mean"] = np.load(models / "oof_mean.npy")
print("cols:", [c for c in df.columns if 'conc' in c or c in ('CAS','species','duration')][:20])
print(df[["CAS", "species", "duration", "conc"]].head())

# ---- within-triplet (pure replicate) variance per chemical -------------
g = df.groupby(["CAS", "species", "duration"], observed=True)["conc"]
trip = pd.DataFrame({"n": g.size(), "var": g.var(ddof=1)}).reset_index()
rep = trip[trip["n"] >= 3]  # need >=3 for a semi-usable variance
# pooled within-triplet variance per chemical (sum SS / sum df)
rep = rep.assign(ss=lambda d: d["var"] * (d["n"] - 1), dof=lambda d: d["n"] - 1)
chem_within = rep.groupby("CAS", observed=True).agg(
    ss=("ss", "sum"), dof=("dof", "sum"), n_trip_rep=("n", "size")).reset_index()
chem_within["within_var"] = chem_within["ss"] / chem_within["dof"]

chem = df.groupby("CAS", observed=True).agg(
    aleatoric_var=("aleatoric_var", "mean"),
    epistemic_var=("epistemic_var", "mean"),
    n_obs=("conc", "size"),
    n_species=("species", "nunique"),
).reset_index()
chem["n_class"] = df.groupby("CAS", observed=True)["tax_class"].nunique().values if "tax_class" in df.columns else np.nan
chem = chem.merge(chem_within[["CAS", "within_var", "dof", "n_trip_rep"]], on="CAS", how="left")

BUCKETS = [(3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 249), (250, None)]
LAB = ["3-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]
cc = chem[chem["n_obs"] >= 3].copy()

print("\n=== alpha_c vs pure within-triplet replicate variance, by n_obs bin ===")
print(f"{'bin':>9} {'Nchem':>6} {'medAle':>8} {'medWithin':>10} {'ratio':>7} "
      f"{'meanAle':>8} {'meanWithin':>11} {'medNspec':>9} {'Nwith_within':>12}")
rows = []
for (lo, hi), lab in zip(BUCKETS, LAB):
    m = (cc["n_obs"] >= lo) & ((cc["n_obs"] <= hi) if hi else True)
    s = cc[m]
    sw = s.dropna(subset=["within_var"])
    r = {"bin": lab, "n_chem": len(s),
         "med_aleatoric_var": s["aleatoric_var"].median(),
         "med_within_var": sw["within_var"].median() if len(sw) else np.nan,
         # means as well as medians: Figure "variance composition" in the paper
         # plots bin means, so the manuscript quotes the mean columns for the
         # aleatoric-vs-replicate comparison and the medians elsewhere.
         "mean_aleatoric_var": s["aleatoric_var"].mean(),
         "mean_within_var": sw["within_var"].mean() if len(sw) else np.nan,
         "med_n_species": s["n_species"].median(),
         "n_chem_with_within": len(sw)}
    r["ratio_ale_over_within"] = r["med_aleatoric_var"] / r["med_within_var"] if len(sw) else np.nan
    rows.append(r)
    print(f"{lab:>9} {len(s):>6} {r['med_aleatoric_var']:>8.4f} "
          f"{r['med_within_var']:>10.4f} {r['ratio_ale_over_within']:>7.2f} "
          f"{r['mean_aleatoric_var']:>8.4f} {r['mean_within_var']:>11.4f} "
          f"{r['med_n_species']:>9.0f} {len(sw):>12}")
pd.DataFrame(rows).to_csv(str(Path(__file__).resolve().parent / "ale_vs_within.csv"), index=False)

# ---- who is in the 250+ bin ----
top = cc[cc["n_obs"] >= 250].sort_values("aleatoric_var", ascending=False)
namecol = [c for c in df.columns if "name" in c.lower()][:5]
print("\nname-ish columns:", namecol)
if namecol:
    nm = df.groupby("CAS", observed=True)[namecol[0]].first()
    top = top.assign(name=top["CAS"].map(nm))
print("\n=== 250+ bin chemicals (top 20 by aleatoric var) ===")
print(top.head(20).to_string(index=False))
print(f"\n250+ bin: n={len(top)}, median n_species={top['n_species'].median():.0f}, "
      f"median aleatoric={top['aleatoric_var'].median():.4f}, "
      f"median within={top['within_var'].median():.4f}")

# ---- global: aleatoric vs within, all chemicals with decent replicate info ----
good = cc.dropna(subset=["within_var"])
good = good[good["dof"] >= 10]
print(f"\nChemicals with >=10 replicate dof: {len(good)}")
print(f"  median aleatoric var {good['aleatoric_var'].median():.4f}, "
      f"median within-triplet var {good['within_var'].median():.4f}, "
      f"median ratio {(good['aleatoric_var']/good['within_var']).median():.2f}")
print(f"  spearman(n_species, aleatoric): "
      f"{good[['n_species','aleatoric_var']].corr(method='spearman').iloc[0,1]:.3f}")
print(f"  spearman(n_species, within): "
      f"{good[['n_species','within_var']].corr(method='spearman').iloc[0,1]:.3f}")
print(f"  spearman(n_obs, aleatoric): "
      f"{good[['n_obs','aleatoric_var']].corr(method='spearman').iloc[0,1]:.3f}")

# ---- b0 chain autocorrelation / ESS ----
print("\n=== b0 chain diagnostics ===")
b0 = np.load(models / "b0_trace.npy")
print("shape", b0.shape)


def ess(x):
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    # Geyer initial positive sequence
    s, t = 0.0, 1
    while t + 1 < n:
        p = ac[t] + ac[t + 1]
        if p < 0:
            break
        s += p
        t += 2
    tau = 1 + 2 * s
    return n / tau, tau, ac


for f in range(b0.shape[0]):
    e, tau, ac = ess(b0[f])
    print(f"  fold {f}: n={b0.shape[1]} ESS={e:.0f} tau_int={tau:.1f} "
          f"lag1={ac[1]:.3f} lag5={ac[5]:.3f} lag20={ac[20]:.3f}")
