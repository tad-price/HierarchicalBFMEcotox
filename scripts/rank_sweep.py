"""
rank_sweep.py - How the uncertainty decomposition responds to model capacity

Refits the Hierarchical BFM under the same grouped cross-validation used in
train_bfm.py, at several factorization ranks k, and records for each rank:

- out-of-fold RMSE,
- epistemic and aleatoric SD, each as sqrt(mean variance) to match the
  convention used for the figures quoted in the main text,
- the aleatoric share of total predictive variance,
- the ratio of the learned aleatoric variance to the *pooled within-triplet*
  replicate variance of the same chemical.

The last of these is the point of the experiment. Within a (chemical, species,
duration) triplet the model predicts one value for every replicate, so the
scatter of those replicates is measurement noise that no amount of model
capacity can explain. If alpha_c were absorbing model misfit, that ratio would
exceed one at low k and fall towards one as k grows, which is what separates a
genuine noise term from an artefact of the chosen rank.

All ranks share identical CV splits (GroupKFold on the triplet id is
deterministic), so differences between ranks are not split effects.

Outputs to outputs/rank_sweep/:
- rank_sweep_results.csv     one row per rank
- rank_sweep.png             RMSE, the two components, and the replicate ratio
- oof_summary_k{K}.npz       per-observation mean/epistemic/aleatoric per rank

Runtime scales as roughly (6.6 + 0.16k) seconds per Gibbs iteration per fold on
an 80%-of-data training partition, so the full sweep is an overnight job. The
results CSV is rewritten after every rank, so a partial run is still usable.

Usage:
    python scripts/rank_sweep.py --k_values 4,8,16,32,64 --n_folds 5 \
        --n_iter 200 --n_burn 50

Arguments mirror train_bfm.py. See --help.
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from model.hierarchical_bfm import HierarchicalBFM, estimate_alpha_a0
from data.load_ecotox import load_ecotox_data
from train_bfm import build_design

OUT_DIR = ROOT_DIR / "outputs" / "rank_sweep"


def effective_sample_size(x):
    """Geyer initial-positive-sequence ESS for a 1-D trace."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return np.nan
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    if ac[0] <= 0:
        return np.nan
    ac /= ac[0]
    total, t = 0.0, 1
    while t + 1 < n:
        pair = ac[t] + ac[t + 1]
        if pair < 0:
            break
        total += pair
        t += 2
    return n / (1.0 + 2.0 * total)


def pooled_within_triplet_variance(df):
    """Replicate variance per chemical, pooled over its triplets by dof.

    Returns a DataFrame indexed by CAS with columns [within_var, dof].
    Only triplets with at least 3 replicates contribute.
    """
    g = df.groupby(["CAS", "species", "duration"], observed=True)["conc"]
    trip = pd.DataFrame({"n": g.size(), "var": g.var(ddof=1)}).reset_index()
    trip = trip[(trip["n"] >= 3) & trip["var"].notna()]
    trip["ss"] = trip["var"] * (trip["n"] - 1)
    trip["dof"] = trip["n"] - 1
    out = trip.groupby("CAS", observed=True).agg(ss=("ss", "sum"), dof=("dof", "sum"))
    out["within_var"] = out["ss"] / out["dof"]
    return out[["within_var", "dof"]]


def make_figure(res):
    """Draw the three-panel rank-sweep figure from the results table."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    C_EPI, C_ALE = "#3b7dba", "#d1495b"

    ax = axes[0]
    ax.plot(res["k"], res["rmse"], "o-", color="#08306b", linewidth=1.8)
    ax.set_xscale("log", base=2); ax.set_xticks(res["k"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Factorization rank $k$"); ax.set_ylabel("Out-of-fold RMSE (log mg/L)")
    ax.set_title("(a) Predictive accuracy"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(res["k"], res["epistemic_sd_rms"], "o-", color=C_EPI,
            linewidth=1.8, label="Epistemic")
    ax.plot(res["k"], res["aleatoric_sd_rms"], "s-", color=C_ALE,
            linewidth=1.8, label="Aleatoric")
    ax.set_xscale("log", base=2); ax.set_xticks(res["k"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Factorization rank $k$"); ax.set_ylabel(r"Predictive SD, $\sqrt{\mathrm{mean\ variance}}$ (log mg/L)")
    ax.set_title("(b) Uncertainty components"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(res["k"], res["ale_over_replicate_var"], "o-", color=C_ALE, linewidth=1.8)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.2,
               label="agreement with replicate noise")
    ax.set_xscale("log", base=2); ax.set_xticks(res["k"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Factorization rank $k$")
    ax.set_ylabel("Aleatoric variance / replicate variance")
    ax.set_title("(c) Is the noise term absorbing misfit?")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "rank_sweep.png", dpi=150)
    plt.close()
    print(f"Saved: {OUT_DIR / 'rank_sweep.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k_values", type=str, default="4,8,16,32,64",
                    help="Comma-separated factorization ranks (default: 4,8,16,32,64)")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Cross-validation folds (default: 5)")
    ap.add_argument("--n_iter", type=int, default=200,
                    help="Gibbs iterations per fold (default: 200)")
    ap.add_argument("--n_burn", type=int, default=50,
                    help="Burn-in iterations (default: 50)")
    ap.add_argument("--eb_mode", choices=["once", "per_k", "fixed"], default="once",
                    help="How to set the noise-prior shape a0. 'once' runs the EB "
                         "pre-pass one time at --eb_k and reuses it for every rank, "
                         "so the noise prior is held constant while k varies "
                         "(default). 'per_k' re-estimates a0 at each rank. 'fixed' "
                         "uses --a0_fixed.")
    ap.add_argument("--eb_k", type=int, default=32,
                    help="Rank used for the one-off EB pre-pass under --eb_mode once "
                         "(default: 32, matching the paper's main run)")
    ap.add_argument("--eb_n_iter", type=int, default=100)
    ap.add_argument("--eb_n_burn", type=int, default=50)
    ap.add_argument("--eb_min_n", type=int, default=50)
    ap.add_argument("--a0_fixed", type=float, default=2.0)
    ap.add_argument("--min_dof", type=int, default=10,
                    help="Min pooled replicate dof for a chemical to enter the "
                         "aleatoric-vs-replicate comparison (default: 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_cache", action="store_true",
                    help="Skip all model fitting and rebuild the results table and "
                         "figure from existing outputs/rank_sweep/oof_summary_k*.npz. "
                         "median_ess and minutes are carried over from the existing "
                         "CSV where available.")
    args = ap.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rank sweep")
    print(f"  ranks      : {k_values}")
    print(f"  folds      : {args.n_folds}")
    print(f"  iterations : {args.n_iter} ({args.n_burn} burn-in, "
          f"{args.n_iter - args.n_burn} retained)")
    print(f"  a0 mode    : {args.eb_mode}")

    # ---- data, encoders, groups (identical to train_bfm.py) ----------------
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False, use_selfies=False, use_mol2vec=False,
        use_fingerprint=False, shuffle=True, random_state=42,
    )
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"] = np.log(full_data["chem_mw"])
    num_cols = ["chem_mw", "chem_rdkit_clogp"]

    enc_dict = {c: sk_prep.OneHotEncoder(handle_unknown="ignore")
                for c in ["species", "CAS", "duration", "tax_family", "tax_class"]}
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    unique_cas = full_data["CAS"].unique()
    cas_to_idx = {cas: i for i, cas in enumerate(unique_cas)}
    groups = full_data["CAS"].map(cas_to_idx).values.astype(int)
    n_groups = len(unique_cas)
    n_per_group = np.bincount(groups, minlength=n_groups)

    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str))[0]
    splits = list(sk_model.GroupKFold(n_splits=args.n_folds)
                  .split(full_data, y_centered, groups=triplet_id))

    within = pooled_within_triplet_variance(full_data)
    within = within[within["dof"] >= args.min_dof]
    print(f"  chemicals with >= {args.min_dof} replicate dof: {len(within)}")

    # ---- noise-prior shape -------------------------------------------------
    def run_eb(k):
        X_full, _, _ = build_design(full_data, enc_dict, num_cols)
        pre = HierarchicalBFM(n_features=X_full.shape[1], n_groups=n_groups, k=k,
                              alpha_a0=1.0, alpha_b0_init=1.0, learn_alpha_b0=False)
        pre.fit(X_full, y_centered, groups=groups,
                n_iter=args.eb_n_iter, n_burn=args.eb_n_burn,
                random_state=args.seed)
        alpha_post = np.stack([s["alpha_vec"] for s in pre.samples], axis=1)
        return float(estimate_alpha_a0(alpha_post, n_per_group, min_n=args.eb_min_n))

    a0_shared = None
    if args.eb_mode == "once":
        print(f"\nEB pre-pass at k={args.eb_k} (shared across all ranks)...")
        a0_shared = run_eb(args.eb_k)
        print(f"  a0 = {a0_shared:.4f}")
    elif args.eb_mode == "fixed":
        a0_shared = args.a0_fixed
        print(f"\nUsing fixed a0 = {a0_shared}")

    # ---- rebuild from cached per-observation arrays ------------------------
    if args.from_cache:
        prev = {}
        csv_path = OUT_DIR / "rank_sweep_results.csv"
        if csv_path.exists():
            prev = pd.read_csv(csv_path).set_index("k").to_dict("index")
        rows = []
        for k in k_values:
            f = OUT_DIR / f"oof_summary_k{k}.npz"
            if not f.exists():
                print(f"  skipping k={k}: {f.name} not found")
                continue
            d = np.load(f)
            oof_mean, oof_epi, oof_ale = (d["oof_mean"], d["oof_epistemic"],
                                          d["oof_aleatoric"])
            chem = pd.DataFrame({"CAS": full_data["CAS"].values, "ale": oof_ale}) \
                     .groupby("CAS", observed=True)["ale"].mean()
            cmp = within.join(chem.rename("ale_var"), how="inner")
            rows.append({
                "k": k,
                "a0": prev.get(k, {}).get("a0", np.nan),
                "rmse": round(float(np.sqrt(np.mean((oof_mean - y_centered) ** 2))), 4),
                "epistemic_sd_rms": round(float(np.sqrt(oof_epi.mean())), 4),
                "aleatoric_sd_rms": round(float(np.sqrt(oof_ale.mean())), 4),
                "aleatoric_share": round(float(oof_ale.mean() /
                                               (oof_ale.mean() + oof_epi.mean())), 4),
                "ale_over_replicate_var": round(
                    float((cmp["ale_var"] / cmp["within_var"]).median()), 4),
                "n_chem_compared": int(len(cmp)),
                "median_ess": prev.get(k, {}).get("median_ess", np.nan),
                "n_retained": prev.get(k, {}).get("n_retained", np.nan),
                "minutes": prev.get(k, {}).get("minutes", np.nan),
            })
        res = pd.DataFrame(rows)
        res.to_csv(OUT_DIR / "rank_sweep_results.csv", index=False)
        print("\n" + res.to_string(index=False))
        make_figure(res)
        return

    # ---- sweep -------------------------------------------------------------
    rows = []
    for k in k_values:
        t0 = time.time()
        if args.eb_mode == "per_k":
            print(f"\nEB pre-pass at k={k}...")
            a0 = run_eb(k)
            print(f"  a0 = {a0:.4f}")
        else:
            a0 = a0_shared

        print(f"\n{'='*60}\nRANK k = {k}\n{'='*60}")
        n_obs = len(full_data)
        oof_mean = np.zeros(n_obs)
        oof_epi = np.zeros(n_obs)
        oof_ale = np.zeros(n_obs)
        ess_vals = []
        rng = np.random.default_rng(args.seed)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
            y_tr = y_centered[tr_idx]

            X_tr, imputer, scaler = build_design(df_tr, enc_dict, num_cols)
            X_va, _, _ = build_design(df_va, enc_dict, num_cols, imputer, scaler)

            model = HierarchicalBFM(n_features=X_tr.shape[1], n_groups=n_groups,
                                    k=k, alpha_a0=a0, alpha_b0_init=1.0,
                                    learn_alpha_b0=True)
            model.fit(X_tr, y_tr, groups=groups[tr_idx],
                      n_iter=args.n_iter, n_burn=args.n_burn,
                      random_state=args.seed + fold)

            X2_va = X_va.copy(); X2_va.data **= 2
            groups_va = groups[va_idx]
            preds, ale = [], []
            for s in model.samples:
                q = X_va @ s["v"]
                inter = 0.5 * ((q ** 2) - X2_va @ (s["v"] ** 2)).sum(axis=1)
                preds.append(s["w0"] + X_va @ s["w"] + inter)
                ale.append(1.0 / s["alpha_vec"][groups_va])
            preds = np.asarray(preds)          # (S, N_va)
            ale = np.asarray(ale)

            oof_mean[va_idx] = preds.mean(axis=0)
            oof_epi[va_idx] = preds.var(axis=0)
            oof_ale[va_idx] = ale.mean(axis=0)

            # ESS on a random subset of prediction traces, to show whether the
            # chain is long enough for the epistemic variance to be trusted.
            probe = rng.choice(preds.shape[1], size=min(100, preds.shape[1]),
                               replace=False)
            ess_vals += [effective_sample_size(preds[:, j]) for j in probe]

            fold_rmse = float(np.sqrt(np.mean(
                (oof_mean[va_idx] - y_centered[va_idx]) ** 2)))
            print(f"  fold {fold+1}/{args.n_folds}  RMSE {fold_rmse:.4f}")

        rmse = float(np.sqrt(np.mean((oof_mean - y_centered) ** 2)))

        # aleatoric vs pooled within-triplet replicate variance, per chemical
        chem = pd.DataFrame({"CAS": full_data["CAS"].values,
                             "ale": oof_ale}).groupby("CAS", observed=True)["ale"].mean()
        cmp = within.join(chem.rename("ale_var"), how="inner")
        ratio = (cmp["ale_var"] / cmp["within_var"]).median()

        ess_med = float(np.nanmedian(ess_vals))
        # SDs are reported as sqrt(mean variance), matching the convention used
        # for the epistemic and aleatoric figures quoted in the main text.
        row = {
            "k": k,
            "a0": round(a0, 4),
            "rmse": round(rmse, 4),
            "epistemic_sd_rms": round(float(np.sqrt(oof_epi.mean())), 4),
            "aleatoric_sd_rms": round(float(np.sqrt(oof_ale.mean())), 4),
            "aleatoric_share": round(float(oof_ale.mean() /
                                           (oof_ale.mean() + oof_epi.mean())), 4),
            "ale_over_replicate_var": round(float(ratio), 4),
            "n_chem_compared": int(len(cmp)),
            "median_ess": round(ess_med, 1),
            "n_retained": args.n_iter - args.n_burn,
            "minutes": round((time.time() - t0) / 60.0, 1),
        }
        rows.append(row)
        np.savez_compressed(OUT_DIR / f"oof_summary_k{k}.npz",
                            oof_mean=oof_mean, oof_epistemic=oof_epi,
                            oof_aleatoric=oof_ale)
        print(f"  -> RMSE {row['rmse']}  epi {row['mean_epistemic_sd']}  "
              f"ale {row['mean_aleatoric_sd']}  ale/replicate "
              f"{row['ale_over_replicate_var']}  ESS {row['median_ess']}  "
              f"({row['minutes']} min)")

        pd.DataFrame(rows).to_csv(OUT_DIR / "rank_sweep_results.csv", index=False)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(res.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'rank_sweep_results.csv'}")

    make_figure(res)



if __name__ == "__main__":
    main()
