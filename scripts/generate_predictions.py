"""
generate_predictions.py - Generate predictions for all (chemical, species, duration) triplets

This script trains the Hierarchical BFM on the entire dataset (no cross-validation)
and generates predictions with uncertainty estimates for all possible triplets.

MEMORY-EFFICIENT: Processes one chemical at a time to avoid OOM errors.

Outputs to outputs/models/:
- trained_model.pkl: Trained model with encoders and metadata
- full_predictions.parquet: Predictions for all triplets

Usage:
    python scripts/generate_predictions.py [--n_iter N] [--n_burn N]

Arguments:
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
from tqdm.auto import tqdm
import gc

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from model.hierarchical_bfm import HierarchicalBFM
from data.load_ecotox import load_ecotox_data
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute


def make_design_cats(df: pd.DataFrame, enc_dict: dict) -> sp.csr_matrix:
    """Create categorical design matrix from dataframe."""
    Xi = enc_dict["species"].transform(df[["species"]])
    Xj = enc_dict["CAS"].transform(df[["CAS"]])
    Xd = enc_dict["duration"].transform(df[["duration"]])
    Xt = enc_dict["tax_family"].transform(df[["tax_family"]])
    Xe = enc_dict["tax_class"].transform(df[["tax_class"]])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate full predictions for all (chemical, species, duration) triplets"
    )
    parser.add_argument(
        "--n_iter", type=int, default=200,
        help="Number of Gibbs sampling iterations (default: 200)"
    )
    parser.add_argument(
        "--n_burn", type=int, default=100,
        help="Number of burn-in iterations (default: 100)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING FULL PREDICTIONS (Memory-Efficient)")
    print("=" * 60)
    print(f"  Iterations: {args.n_iter}, Burn-in: {args.n_burn}")
    
    # 1. Load Data (same as train_bfm.py)
    print("\n1. Loading data...")
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
    
    print(f"   Loaded {len(full_data):,} observations")
    print(f"   Unique chemicals: {full_data['CAS'].nunique()}")
    print(f"   Unique species: {full_data['species'].nunique()}")
    print(f"   Unique durations: {full_data['duration'].nunique()}")

    # 2. Prepare Encoders
    print("\n2. Preparing encoders...")
    enc_dict = {
        "species":     sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "CAS":         sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "duration":    sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_family":  sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_class":   sk_prep.OneHotEncoder(handle_unknown="ignore"),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # 3. Prepare Groups (CAS indices)
    unique_cas = full_data["CAS"].unique()
    cas_to_idx = {cas: i for i, cas in enumerate(unique_cas)}
    groups = full_data["CAS"].map(cas_to_idx).values.astype(int)
    n_groups = len(unique_cas)
    print(f"   Number of groups (chemicals): {n_groups}")

    # 4. Prepare training data (full dataset)
    print("\n3. Preparing training data...")
    X_cat = make_design_cats(full_data, enc_dict)
    
    imputer = sk_impute.SimpleImputer(strategy="median")
    scaler  = sk_prep.StandardScaler()
    num_train = scaler.fit_transform(imputer.fit_transform(full_data[num_cols]))
    
    X_train = sp.hstack([X_cat, sp.csr_matrix(num_train)], format="csr")
    y_train = y_centered
    
    print(f"   X shape: {X_train.shape}")

    # 5. Train model on full dataset
    print("\n4. Training model on full dataset...")
    print(f"   (k=32, n_iter={args.n_iter}, n_burn={args.n_burn})")
    
    model = HierarchicalBFM(n_features=X_train.shape[1], n_groups=n_groups, k=32)
    model.fit(X_train, y_train, groups=groups, n_iter=args.n_iter, n_burn=args.n_burn)
    
    print(f"   Collected {len(model.samples)} posterior samples")

    # 5b. Save trained model IMMEDIATELY (before prediction grid, in case of crash)
    OUTPUTS = ROOT_DIR / "outputs" / "models"
    OUTPUTS.mkdir(exist_ok=True, parents=True)
    
    model_path = OUTPUTS / "trained_model.pkl"
    print(f"\n   Saving trained model to: {model_path}")
    
    unique_species = full_data["species"].unique()
    unique_durations = full_data["duration"].unique()
    
    # Pre-compute lookup tables for species taxonomy and chemical properties
    species_to_tax = full_data.groupby("species", observed=True)[["tax_family", "tax_class"]].first()
    chem_props = full_data.groupby("CAS", observed=True)[num_cols].first()
    
    joblib.dump({
        "model": model,
        "enc_dict": enc_dict,
        "imputer": imputer,
        "scaler": scaler,
        "cas_to_idx": cas_to_idx,
        "unique_cas": unique_cas,
        "unique_species": unique_species,
        "unique_durations": unique_durations,
        "num_cols": num_cols,
        "species_to_tax": species_to_tax,
        "chem_props": chem_props,
        "y_mean": y_mean,
    }, model_path)
    print(f"   Model saved successfully!")

    # 6. Generate predictions ONE CHEMICAL AT A TIME (memory-efficient)
    print("\n5. Generating predictions (one chemical at a time)...")
    
    n_triplets_per_chem = len(unique_species) * len(unique_durations)
    print(f"   Triplets per chemical: {n_triplets_per_chem:,}")
    print(f"   Total chemicals: {len(unique_cas):,}")
    
    output_path = OUTPUTS / "full_predictions.parquet"
    
    # Process chemicals in batches and collect results
    all_results = []
    
    for cas_idx, cas in enumerate(tqdm(unique_cas, desc="Processing chemicals")):
        # Build mini-grid for this chemical
        rows = []
        for species in unique_species:
            for duration in unique_durations:
                rows.append({
                    "CAS": cas,
                    "species": species,
                    "duration": duration
                })
        
        grid_df = pd.DataFrame(rows)
        grid_df["duration"] = pd.Categorical(grid_df["duration"], categories=unique_durations)
        
        # Add taxonomy info
        grid_df = grid_df.merge(species_to_tax, on="species", how="left")
        grid_df["tax_family"] = grid_df["tax_family"].fillna("Unknown")
        grid_df["tax_class"] = grid_df["tax_class"].fillna("Unknown")
        
        # Add chemical properties
        grid_df = grid_df.merge(chem_props, on="CAS", how="left")
        
        # Generate design matrix
        X_grid_cat = make_design_cats(grid_df, enc_dict)
        num_grid = scaler.transform(imputer.transform(grid_df[num_cols]))
        X_grid = sp.hstack([X_grid_cat, sp.csr_matrix(num_grid)], format="csr")
        X2_grid = X_grid.copy(); X2_grid.data **= 2
        
        # Chemical group index for this CAS (all rows have same group)
        chem_group_idx = cas_to_idx[cas]
        
        # Predict using posterior samples
        n_samples = len(model.samples)
        preds_all = np.zeros((n_samples, len(grid_df)))
        aleatoric_all = np.zeros(n_samples)  # Same for all rows in this chemical
        
        for i, s in enumerate(model.samples):
            w0, w, v, alpha_vec = s["w0"], s["w"], s["v"], s["alpha_vec"]
            
            # Prediction
            q = X_grid @ v
            inter = 0.5 * ((q**2) - X2_grid @ (v**2)).sum(axis=1)
            preds_all[i] = w0 + X_grid @ w + inter
            
            # Aleatoric variance (same for all rows - this chemical's alpha)
            aleatoric_all[i] = 1.0 / alpha_vec[chem_group_idx]
        
        # Aggregate
        pred_mean = preds_all.mean(axis=0)
        pred_epistemic_var = preds_all.var(axis=0)
        pred_aleatoric_var = aleatoric_all.mean()  # Single value for this chemical
        
        # Store results
        result_df = grid_df[["CAS", "species", "duration"]].copy()
        result_df["pred_mean"] = pred_mean + y_mean
        result_df["pred_epistemic_var"] = pred_epistemic_var
        result_df["pred_aleatoric_var"] = pred_aleatoric_var
        
        all_results.append(result_df)
        
        # Periodic garbage collection and progress
        if (cas_idx + 1) % 100 == 0:
            gc.collect()
    
    # Combine all results
    print("\n6. Combining results...")
    output_df = pd.concat(all_results, ignore_index=True)
    output_df["pred_total_var"] = output_df["pred_epistemic_var"] + output_df["pred_aleatoric_var"]
    output_df["pred_total_sd"] = np.sqrt(output_df["pred_total_var"])
    
    # Save as parquet
    print("\n7. Saving results...")
    output_df.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Total rows: {len(output_df):,}")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Predictions mean range: [{output_df['pred_mean'].min():.2f}, {output_df['pred_mean'].max():.2f}]")
    print(f"Epistemic SD range: [{np.sqrt(output_df['pred_epistemic_var']).min():.3f}, {np.sqrt(output_df['pred_epistemic_var']).max():.3f}]")
    print(f"Aleatoric SD range: [{np.sqrt(output_df['pred_aleatoric_var']).min():.3f}, {np.sqrt(output_df['pred_aleatoric_var']).max():.3f}]")
    print(f"Total SD range: [{output_df['pred_total_sd'].min():.3f}, {output_df['pred_total_sd'].max():.3f}]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
