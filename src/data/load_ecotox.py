"""
Data loading utilities for ecotoxicology dataset.

Loads and preprocesses the ECOTOX mortality data with chemical properties.
"""

import pandas as pd


def load_ecotox_data(
    adore_path,
    chemicals_path,
    use_molar=True,
    mol_col="result_conc1_mean_mol_log",
    mg_col="result_conc1_mean_log",
    use_selfies=False,
    selfies_path=None,
    selfies_col="selfies_embed",
    use_mol2vec=False,
    mol2vec_path=None,
    mol2vec_cols=None,
    use_fingerprint=False,
    fingerprint_path=None,
    fp_col="morgan_fp",
    shuffle=True,
    random_state=42,
):
    """
    Load and preprocess ecotoxicology data.
    
    Args:
        adore_path: Path to the main ecotox mortality CSV file
        chemicals_path: Path to the chemical properties CSV file
        use_molar: If True, use log molar concentration; if False, use log mg/L
        mol_col: Column name for log molar concentration
        mg_col: Column name for log mg/L concentration
        use_selfies: Whether to include SELFIES embeddings
        selfies_path: Path to SELFIES embeddings file
        selfies_col: Column name for SELFIES embeddings
        use_mol2vec: Whether to include Mol2Vec embeddings
        mol2vec_path: Path to Mol2Vec embeddings file
        mol2vec_cols: Column names for Mol2Vec features
        use_fingerprint: Whether to include molecular fingerprints
        fingerprint_path: Path to fingerprints file
        fp_col: Column name for fingerprints
        shuffle: Whether to shuffle the data
        random_state: Random seed for shuffling
        
    Returns:
        df: Preprocessed DataFrame
        y_centered: Centered concentration values (target variable)
    """
    # 1) Base merges
    adore = pd.read_csv(adore_path, low_memory=False)
    chemicals = pd.read_csv(chemicals_path, low_memory=False)

    df = adore.merge(chemicals, on="test_cas", how="left")
    df.rename(
        columns={
            "tax_gs": "species",
            "test_cas": "CAS",
            "result_obs_duration_mean": "duration",
        },
        inplace=True,
    )

    target_col = mol_col if use_molar else mg_col
    if target_col not in df.columns:
        raise ValueError(f"Chosen target_col='{target_col}' not found.")
    df.rename(columns={target_col: "conc"}, inplace=True)

    if use_selfies:
        if not selfies_path:
            raise ValueError("`use_selfies=True` but `selfies_path` not provided.")
        df_selfies = pd.read_csv(selfies_path, low_memory=False)
        df = df.merge(df_selfies[["CAS", selfies_col]], on="CAS", how="left")

    if use_mol2vec:
        if mol2vec_path is None:
            if not mol2vec_cols:
                raise ValueError("use_mol2vec=True needs mol2vec_cols when mol2vec_path is None.")
            missing_cols = [c for c in mol2vec_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Mol2Vec columns missing in chemicals file: {missing_cols}")
        else:
            if not mol2vec_cols:
                raise ValueError("`use_mol2vec=True` requires mol2vec_cols.")
            df_m2v = pd.read_csv(mol2vec_path, low_memory=False)
            need = ["CAS"] + mol2vec_cols
            miss = [c for c in need if c not in df_m2v.columns]
            if miss:
                raise ValueError(f"Mol2Vec file missing columns: {miss}")
            df = df.merge(df_m2v[need], on="CAS", how="left")

    if use_fingerprint:
        if not fingerprint_path:
            raise ValueError("`use_fingerprint=True` but `fingerprint_path` not provided.")
        df_fp = pd.read_csv(
            fingerprint_path,
            usecols=["CAS", fp_col],
            dtype={"CAS": "string", fp_col: "string"},
            low_memory=False,
        )
        fp_map = df_fp.set_index("CAS")[fp_col]
        df[fp_col] = df["CAS"].astype("string").map(fp_map)

    df["species"] = pd.Categorical(df["species"])
    df["CAS"] = pd.Categorical(df["CAS"])
    df["duration"] = df["duration"].astype(float)

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_mean = df["conc"].mean()
    df["conc_centered"] = df["conc"] - y_mean

    return df, df["conc_centered"].values
