"""
train_bfm.py - Train Hierarchical BFM with Cross-Validation

This script trains the Hierarchical Bayesian Factorization Machine using k-fold
cross-validation and saves out-of-fold predictions with uncertainty estimates.

Outputs to outputs/models/:
- oof_mean.npy: Out-of-fold mean predictions
- oof_epistemic.npy: Out-of-fold epistemic variance
- oof_aleatoric.npy: Out-of-fold aleatoric variance
- oof_pred_samples.npy: Per-sample OOF predictions, shape (N, S)
- oof_aleatoric_samples.npy: Per-sample OOF aleatoric variances (1/alpha_c), shape (N, S)

Usage:
    python scripts/train_bfm.py [--n_folds N] [--n_iter N] [--n_burn N]

Arguments:
    --n_folds   Number of cross-validation folds (default: 3)
    --n_iter    Number of Gibbs sampling iterations (default: 200)
    --n_burn    Number of burn-in iterations (default: 100)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from model.hierarchical_bfm import HierarchicalBFM, estimate_alpha_a0
from data.load_ecotox import load_ecotox_data
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute
import sklearn.model_selection as sk_model


def make_design_cats(df: pd.DataFrame, enc_dict: dict) -> sp.csr_matrix:
    """Create categorical design matrix from dataframe."""
    Xi = enc_dict["species"].transform(df[["species"]])
    Xj = enc_dict["CAS"].transform(df[["CAS"]])
    Xd = enc_dict["duration"].transform(df[["duration"]])
    Xt = enc_dict["tax_family"].transform(df[["tax_family"]])
    Xe = enc_dict["tax_class"].transform(df[["tax_class"]])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")


def build_design(df: pd.DataFrame, enc_dict: dict, num_cols: list,
                 imputer=None, scaler=None):
    """Build full design matrix (cats + numerics). Fits imputer/scaler if not
    provided, otherwise transforms with the given ones. Returns (X, imputer, scaler)."""
    X_cat = make_design_cats(df, enc_dict)
    if imputer is None:
        imputer = sk_impute.SimpleImputer(strategy="median")
        num = imputer.fit_transform(df[num_cols])
    else:
        num = imputer.transform(df[num_cols])
    if scaler is None:
        scaler = sk_prep.StandardScaler()
        num = scaler.fit_transform(num)
    else:
        num = scaler.transform(num)
    X = sp.hstack([X_cat, sp.csr_matrix(num)], format="csr")
    return X, imputer, scaler


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train Hierarchical BFM with Cross-Validation"
    )
    parser.add_argument(
        "--n_folds", type=int, default=3,
        help="Number of cross-validation folds (default: 3)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=200,
        help="Number of Gibbs sampling iterations (default: 200)"
    )
    parser.add_argument(
        "--n_burn", type=int, default=100,
        help="Number of burn-in iterations (default: 100)"
    )
    parser.add_argument(
        "--eb_a0", action=argparse.BooleanOptionalAction, default=True,
        help="Empirical-Bayes estimate alpha_a0 from a pre-pass (default: True). "
             "Use --no-eb_a0 to disable and use --a0_fixed."
    )
    parser.add_argument(
        "--eb_n_iter", type=int, default=100,
        help="EB pre-pass Gibbs iterations (default: 100)"
    )
    parser.add_argument(
        "--eb_n_burn", type=int, default=50,
        help="EB pre-pass burn-in (default: 50)"
    )
    parser.add_argument(
        "--eb_min_n", type=int, default=50,
        help="Min N_c for a chemical to be counted as data-rich in EB fit (default: 50)"
    )
    parser.add_argument(
        "--a0_fixed", type=float, default=2.0,
        help="Fallback alpha_a0 when --no-eb_a0 (default: 2.0). Avoid 1.0."
    )
    parser.add_argument(
        "--learn_b0", action=argparse.BooleanOptionalAction, default=True,
        help="Learn alpha_b0 via Gamma hyperprior (default: True)."
    )
    args = parser.parse_args()
    
    print("Running Hierarchical BFM Experiment...")
    print(f"  Folds: {args.n_folds}, Iterations: {args.n_iter}, Burn-in: {args.n_burn}")
    
    # 1. Load Data (using mg/L units for interpretability)
    DATA_DIR = ROOT_DIR / "data" / "raw"
    full_data, y_centered, y_mean = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_molar=False,  # Use log mg/L instead of log molar
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"]  = np.log(full_data["chem_mw"])
    num_cols = ["chem_mw", "chem_rdkit_clogp"]

    # 2. Prepare Encoders
    enc_dict = {
        "species":     sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "CAS":         sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "duration":    sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_family":  sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_class":   sk_prep.OneHotEncoder(handle_unknown="ignore"),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # 3. Prepare Groups (CAS indices) for Hierarchical Model
    # Map each CAS string to an integer index 0..N_groups-1
    unique_cas = full_data["CAS"].unique()
    cas_to_idx = {cas: i for i, cas in enumerate(unique_cas)}
    groups = full_data["CAS"].map(cas_to_idx).values.astype(int)
    n_groups = len(unique_cas)
    print(f"Number of groups (chemicals): {n_groups}")

    # 3b. EB pre-pass: estimate alpha_a0 from a quick fit on full training data.
    # The data-rich chemicals' posterior alphas are essentially data-driven, so
    # method-of-moments on them gives a sensible Gamma shape for the prior.
    # alpha_b0 is then learned via its Gamma hyperprior during the CV runs.
    OUTPUTS = ROOT_DIR / "outputs" / "models"
    OUTPUTS.mkdir(exist_ok=True, parents=True)
    n_per_group_full = np.bincount(groups, minlength=n_groups)

    if args.eb_a0:
        print(f"\nEB pre-pass: estimating alpha_a0 "
              f"(n_iter={args.eb_n_iter}, n_burn={args.eb_n_burn}, "
              f"min_n={args.eb_min_n})...")
        X_full, _, _ = build_design(full_data, enc_dict, num_cols)
        pre = HierarchicalBFM(
            n_features=X_full.shape[1], n_groups=n_groups, k=32,
            alpha_a0=1.0, alpha_b0_init=1.0, learn_alpha_b0=False,
        )
        pre.fit(X_full, y_centered, groups=groups,
                n_iter=args.eb_n_iter, n_burn=args.eb_n_burn,
                random_state=42)
        alpha_post = np.stack([s["alpha_vec"] for s in pre.samples], axis=1)  # (G, S)
        a0_hat = estimate_alpha_a0(alpha_post, n_per_group_full,
                                   min_n=args.eb_min_n)
        rich = n_per_group_full >= args.eb_min_n
        print(f"  EB result: a0_hat = {a0_hat:.4f} "
              f"(from {int(rich.sum())} data-rich chemicals; "
              f"mean alpha = {alpha_post[rich].mean(axis=1).mean():.4f}, "
              f"std = {alpha_post[rich].mean(axis=1).std():.4f})")
        np.save(OUTPUTS / "eb_a0_hat.npy", np.array(a0_hat))
        del pre, alpha_post, X_full
    else:
        a0_hat = args.a0_fixed
        print(f"\nUsing fixed alpha_a0 = {a0_hat}")

    # 4. Create CV Splits
    
    print("Creating GroupKFold splits...")
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str)
    )[0]
    gkf = sk_model.GroupKFold(n_splits=args.n_folds)
    splits = list(gkf.split(full_data, y_centered, groups=triplet_id))

    # 5. CV Loop
    n_obs = len(full_data)
    n_samples = args.n_iter - args.n_burn
    oof_mean = np.zeros(n_obs)
    oof_epistemic = np.zeros(n_obs)
    oof_aleatoric = np.zeros(n_obs)
    # Full per-sample arrays for posterior predictive calibration
    oof_pred_samples = np.zeros((n_obs, n_samples), dtype=np.float32)
    oof_aleatoric_samples = np.zeros((n_obs, n_samples), dtype=np.float32)

    # Store learned alphas for analysis (dictionary mapping CAS -> list of alphas across folds)
    cas_alpha_samples = {cas: [] for cas in unique_cas}
    # Trace of learned b_0 across kept samples, one row per fold
    b0_trace = np.zeros((args.n_folds, n_samples), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\nFold {fold+1}/{args.n_folds}")

        
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va   = y_centered[tr_idx], y_centered[va_idx]
        groups_tr    = groups[tr_idx]
        
        # Prepare X
        X_tr, imputer, scaler = build_design(df_tr, enc_dict, num_cols)
        X_va, _, _ = build_design(df_va, enc_dict, num_cols, imputer, scaler)

        # Train Hierarchical BFM with EB-fixed a_0 and learned b_0 (hyperprior)
        model = HierarchicalBFM(
            n_features=X_tr.shape[1], n_groups=n_groups, k=32,
            alpha_a0=a0_hat, alpha_b0_init=1.0,
            learn_alpha_b0=args.learn_b0,
        )
        model.fit(X_tr, y_tr, groups=groups_tr, n_iter=args.n_iter, n_burn=args.n_burn)
        
        # Predict
        X2_va = X_va.copy(); X2_va.data **= 2
        preds = []
        
        groups_va = groups[va_idx]
        aleatoric_samples = []

        for s in model.samples:
            w0, w, v, alpha_vec = s["w0"], s["w"], s["v"], s["alpha_vec"]
            
            # Prediction
            q = X_va @ v
            inter = 0.5 * ((q**2) - X2_va @ (v**2)).sum(axis=1)
            pred = w0 + X_va @ w + inter
            preds.append(pred)
            
            # Aleatoric (1/alpha) for each validation point
            # alpha_vec is size n_groups. groups_va maps val rows to groups.
            alpha_vals = alpha_vec[groups_va]
            aleatoric_samples.append(1.0 / alpha_vals)
            
            # Store alphas for analysis
            for cas, idx in cas_to_idx.items():
                cas_alpha_samples[cas].append(alpha_vec[idx])

        preds = np.array(preds)
        aleatoric_samples = np.array(aleatoric_samples)
        b0_trace[fold] = np.array([s["alpha_b0"] for s in model.samples])
        
        oof_mean[va_idx] = preds.mean(axis=0)
        oof_epistemic[va_idx] = preds.var(axis=0)
        oof_aleatoric[va_idx] = aleatoric_samples.mean(axis=0)
        # Retain full posterior for calibration analysis. preds and
        # aleatoric_samples are (S, N_va); transpose to (N_va, S).
        oof_pred_samples[va_idx] = preds.T.astype(np.float32)
        oof_aleatoric_samples[va_idx] = aleatoric_samples.T.astype(np.float32)
        
        rmse = np.sqrt(np.mean((oof_mean[va_idx] - y_va)**2))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    # 6. Save Results
    np.save(OUTPUTS / "oof_mean.npy", oof_mean)
    np.save(OUTPUTS / "oof_epistemic.npy", oof_epistemic)
    np.save(OUTPUTS / "oof_aleatoric.npy", oof_aleatoric)
    np.save(OUTPUTS / "oof_pred_samples.npy", oof_pred_samples)
    np.save(OUTPUTS / "oof_aleatoric_samples.npy", oof_aleatoric_samples)
    np.save(OUTPUTS / "y_mean.npy", np.array(y_mean))
    np.save(OUTPUTS / "b0_trace.npy", b0_trace)
    
    # Overall RMSE
    overall_rmse = np.sqrt(np.mean((oof_mean - y_centered)**2))
    print(f"\nOverall RMSE: {overall_rmse:.4f}")
    
    # 7. Analysis & Plotting
    FIGURES = ROOT_DIR / "outputs" / "figures"
    FIGURES.mkdir(exist_ok=True, parents=True)
    
    print("Generating plots...")
    df = full_data.copy()
    df["aleatoric_sd"] = np.sqrt(oof_aleatoric)
    
    # Group by CAS
    chem_stats = df.groupby("CAS").agg(
        n_obs=("CAS", "size"),
        mean_aleatoric_sd=("aleatoric_sd", "mean")
    ).reset_index()
    
    # Plot 1: Aleatoric SD vs N_obs
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=chem_stats, x="n_obs", y="mean_aleatoric_sd", alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Number of Observations (log)")
    plt.ylabel("Learned Aleatoric SD")
    plt.title("Hierarchical BFM: Aleatoric Uncertainty vs Data Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES / "aleatoric_vs_nobs.png")
    plt.close()
    
    # Plot 2: Distribution of Aleatoric SD
    plt.figure(figsize=(10, 6))
    sns.histplot(chem_stats["mean_aleatoric_sd"], bins=50)
    plt.xlabel("Aleatoric SD")
    plt.title("Distribution of Per-Chemical Aleatoric Uncertainty")
    plt.savefig(FIGURES / "aleatoric_dist.png")
    plt.close()
    
    print(f"Done! Outputs saved to {OUTPUTS}")


if __name__ == "__main__":
    main()
