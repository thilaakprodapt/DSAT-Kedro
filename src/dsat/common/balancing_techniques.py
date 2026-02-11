"""Data balancing techniques for categorical and continuous targets."""

import pandas as pd
import numpy as np
from typing import List
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import smogn


def run_imbalance_analysis(df: pd.DataFrame, target_column: str):
    """
    Runs imbalance analysis on a dataframe AFTER target cleaning.
    
    Args:
        df: DataFrame with cleaned target column
        target_column: Name of the target column
        
    Returns:
        dict: Contains target_type, is_balanced, and details
    """
    from dsat.common.imbalance_utils import check_imbalance, check_continuous_imbalance
    
    unique_vals = df[target_column].dropna().unique()
    num_unique = len(unique_vals)

    if pd.api.types.is_numeric_dtype(df[target_column]):
        if num_unique <= 10 or set(unique_vals).issubset({0, 1}):
            target_type = "categorical"
        else:
            target_type = "continuous"
    else:
        target_type = "categorical"

    if target_type == "categorical":
        imbalance_result = check_imbalance(df[target_column])
        return {
            "target_type": target_type,
            "is_balanced": imbalance_result["status"] == "balanced",
            "details": imbalance_result
        }
    else:
        imbalance_result = check_continuous_imbalance(df[target_column])
        return {
            "target_type": target_type,
            "is_balanced": not imbalance_result["is_imbalanced"],
            "details": imbalance_result
        }


# -----------------------------
# CATEGORICAL TARGET FUNCTIONS
# -----------------------------

def pandas_random_oversample(df, target_col):
    """Random oversampling for categorical targets."""
    df = df.copy()
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    class_counts = df[target_col].value_counts()
    max_count = class_counts.max()

    balanced_parts = []
    for cls, count in class_counts.items():
        cls_df = df[df[target_col] == cls]
        if count < max_count:
            extra_samples = cls_df.sample(
                n=max_count - count,
                replace=True,
                random_state=42
            )
            cls_df = pd.concat([cls_df, extra_samples], ignore_index=True)
        balanced_parts.append(cls_df)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def pandas_random_undersample(df, target_col):
    """Random undersampling for categorical targets."""
    df = df.copy()
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    class_counts = df[target_col].value_counts()
    min_count = class_counts.min()

    balanced_parts = []
    for cls in class_counts.index:
        cls_df = df[df[target_col] == cls].sample(n=min_count, random_state=42)
        balanced_parts.append(cls_df)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def smote_oversample(df, target_col):
    """SMOTE oversampling for categorical targets with numeric features only."""
    df = df.copy()

    X = df.drop(columns=[target_col]).astype("float64")
    y = df[target_col]

    # Ensure target is integer labels
    if y.dtype != "int64":
        y = y.astype("int64")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=X.columns)

    # Cast encoded categoricals back to int
    int_cols = [
        col for col in df.columns
        if col != target_col and df[col].dtype == "int64"
    ]
    df_res[int_cols] = df_res[int_cols].round().astype("int64")

    # Target back to int
    df_res[target_col] = y_res.astype("int64")

    return df_res


def smotenc_resample(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    random_state: int = 42,
    k_neighbors: int = None
) -> pd.DataFrame:
    """
    Apply SMOTENC to a mixed-type dataset and safely restore dtypes.

    Key points:
    - SMOTENC operates in continuous space → outputs float arrays (expected).
    - Categorical features are encoded before resampling and restored after.
    - Target column is explicitly restored as int64 (0/1), BigQuery-safe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target.
    target_col : str
        Binary target column (0/1).
    categorical_cols : list[str]
        List of categorical feature columns.
    random_state : int
        Random seed.
    k_neighbors : int | None
        Optional k_neighbors override for SMOTENC.

    Returns
    -------
    pd.DataFrame
        Balanced dataframe with restored dtypes.
    """

    # ------------------------------------------------------------------
    # 1. Basic validation
    # ------------------------------------------------------------------
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    df = df.copy()

    # Prevent target leakage
    categorical_cols = [c for c in categorical_cols if c != target_col]

    missing = set(categorical_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Categorical columns missing from dataframe: {missing}")

    # ------------------------------------------------------------------
    # 2. Split features and target
    # ------------------------------------------------------------------
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Ensure numeric target for SMOTENC
    y_num = pd.to_numeric(y, errors="coerce")

    if y_num.isna().any():
        # Encode non-numeric targets safely
        y_cat = pd.Categorical(y)
        valid_mask = y_cat.codes != -1
        X = X.loc[valid_mask].copy()
        y_num = pd.Series(y_cat.codes[valid_mask], index=X.index)

    class_counts = y_num.value_counts()
    if len(class_counts) < 2:
        # Cannot resample single-class data
        return df.copy()

    minority_count = class_counts.min()

    if k_neighbors is None:
        k_neighbors = max(1, min(5, minority_count - 1))

    # ------------------------------------------------------------------
    # 3. Encode categorical features
    # ------------------------------------------------------------------
    X_enc = X.copy()
    category_maps = {}

    for col in categorical_cols:
        cat = pd.Categorical(X[col].astype(object))
        category_maps[col] = cat.categories
        X_enc[col] = cat.codes

    # Indices of categorical columns for SMOTENC
    cat_indices = [X.columns.get_loc(col) for col in categorical_cols]

    # ------------------------------------------------------------------
    # 4. Apply SMOTENC
    # ------------------------------------------------------------------
    smote = SMOTENC(
        categorical_features=cat_indices,
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_np = X_enc.to_numpy(dtype="float64")
    y_np = y_num.to_numpy(dtype="int64")

    X_res, y_res = smote.fit_resample(X_np, y_np)

    # ------------------------------------------------------------------
    # 5. Reconstruct dataframe
    # ------------------------------------------------------------------
    balanced = pd.DataFrame(X_res, columns=X.columns)

    # Restore categorical columns
    for col in categorical_cols:
        balanced[col] = (
            balanced[col]
            .round()
            .astype("int64")
            .map(dict(enumerate(category_maps[col])))
        )

    # Numeric columns remain float (due to interpolation)
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    balanced[numeric_cols] = balanced[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Restore target column as strict int64
    balanced[target_col] = (
        pd.Series(y_res, index=balanced.index)
        .round()
        .astype("int64")
    )

    # ------------------------------------------------------------------
    # 6. Final validation (important for pipelines)
    # ------------------------------------------------------------------
    if not set(balanced[target_col].unique()).issubset({0, 1}):
        raise ValueError(
            f"Target column corrupted after SMOTENC. "
            f"Found values: {balanced[target_col].unique()}"
        )

    return balanced


def adasyn_oversample(df, target_col):
    """ADASYN oversampling for categorical targets."""
    df = df.copy()

    X = df.drop(columns=[target_col]).astype("float64")
    y = df[target_col].astype("int64")

    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=X.columns)

    int_cols = [
        col for col in df.columns
        if col != target_col and df[col].dtype == "int64"
    ]
    df_res[int_cols] = df_res[int_cols].round().astype("int64")

    df_res[target_col] = y_res.astype("int64")

    return df_res


def cluster_based_oversample(df, target_col, n_clusters=5):
    """Cluster-based oversampling using KMeans."""
    df = df.copy()

    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=False)
    y = df[target_col]

    class_counts = y.value_counts()
    max_count = class_counts.max()

    balanced_parts = []

    for cls, count in class_counts.items():
        cls_df = df[df[target_col] == cls]

        if count == max_count:
            balanced_parts.append(cls_df)
            continue

        X_cls = X.loc[cls_df.index]

        k = min(n_clusters, len(cls_df))
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_cls)

        cls_df = cls_df.copy()
        cls_df["__cluster__"] = labels

        samples_per_cluster = int(np.ceil((max_count - count) / k))

        synthetic_parts = []
        for c in range(k):
            cluster_df = cls_df[cls_df["__cluster__"] == c].drop(columns="__cluster__")

            if len(cluster_df) == 0:
                continue

            synthetic = cluster_df.sample(
                n=samples_per_cluster,
                replace=True,
                random_state=42
            )
            synthetic_parts.append(synthetic)

        oversampled_cls = pd.concat(
            [cls_df.drop(columns="__cluster__")] + synthetic_parts,
            ignore_index=True
        ).sample(n=max_count, random_state=42)

        balanced_parts.append(oversampled_cls)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


# -----------------------------
# CONTINUOUS TARGET FUNCTIONS
# -----------------------------

def smoter_smogn(df, target_col):
    """
    Apply SMOTER (via smogn) on a continuous target column safely.
    """
    df = df.copy()
    
    # Ensure target column is numeric and NaN-free
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    # Standardize column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    try:
        balanced_df = smogn.smoter(
            data=df,
            y=target_col
        )
        return balanced_df
    except Exception as e:
        print(f"SMOTER failed: {e}")
        return df.copy()


def gaussian_noise_injection(df, target_col, noise_level=0.05):
    """Add Gaussian noise to continuous target values."""
    df = df.copy()
    values = df[target_col].values
    noise = np.random.normal(0, noise_level * values.std(), size=len(values))
    df[target_col] = values + noise
    return df


def kde_resample(df, target_col, n_samples=None):
    """KDE-based resampling for continuous targets."""
    df = df.copy()
    X = df[target_col].values[:, np.newaxis]

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    n_samples = n_samples or len(df)
    X_resampled = kde.sample(n_samples)
    df_resampled = df.sample(n=0).reset_index(drop=True)
    df_resampled[target_col] = X_resampled.flatten()
    # Keep other columns the same (can duplicate randomly)
    for col in df.columns:
        if col != target_col:
            df_resampled[col] = df[col].sample(n=n_samples, replace=True).values
    return df_resampled


def quantile_binning_oversample(df, target_col, q=10):
    """Quantile-based binning + oversampling."""
    df = df.copy()
    df['bin'] = pd.qcut(df[target_col], q=q, duplicates='drop')
    class_counts = df['bin'].value_counts()
    max_count = class_counts.max()

    balanced_parts = []
    for b in class_counts.index:
        bin_df = df[df['bin'] == b]
        if len(bin_df) < max_count:
            extra_samples = bin_df.sample(n=max_count - len(bin_df), replace=True, random_state=42)
            bin_df = pd.concat([bin_df, extra_samples], ignore_index=True)
        balanced_parts.append(bin_df)

    balanced_df = pd.concat(balanced_parts, ignore_index=True).drop(columns=['bin'])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


def tail_focused_resampling(df, target_col, tail_fraction=0.05):
    """Tail-focused resampling for continuous targets."""
    df = df.copy()
    upper_threshold = df[target_col].quantile(1 - tail_fraction)
    lower_threshold = df[target_col].quantile(tail_fraction)

    tails = df[(df[target_col] <= lower_threshold) | (df[target_col] >= upper_threshold)]
    n_needed = len(df) - len(tails)
    if len(tails) > 0 and n_needed > 0:
        extra_samples = tails.sample(n=n_needed, replace=True, random_state=42)
        df = pd.concat([df, extra_samples], ignore_index=True)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)
