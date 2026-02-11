"""Imbalance detection and data cleaning utilities."""

import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
from google.oauth2 import service_account


# Initialize BigQuery client with service account
def _get_credentials():
    """Get credentials from service account file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        return service_account.Credentials.from_service_account_file(credentials_path)
    return None

credentials = _get_credentials()
bq_client = bigquery.Client(credentials=credentials) if credentials else bigquery.Client()


def check_imbalance(series):
    """Check imbalance for categorical target.
    
    Args:
        series: Pandas series containing categorical target values
        
    Returns:
        dict: Contains status, class_distribution, and minority_ratio
    """
    dist = series.value_counts(normalize=True)
    minority_ratio = dist.min()

    if minority_ratio >= 0.45:
        status = "balanced"
    elif minority_ratio >= 0.30:
        status = "mildly_imbalanced"
    elif minority_ratio >= 0.10:
        status = "moderately_imbalanced"
    else:
        status = "severely_imbalanced"

    return {
        "status": status,
        "class_distribution": dist.to_dict(),
        "minority_ratio": minority_ratio
    }


def check_continuous_imbalance(series, skew_threshold=1.0, tail_threshold=0.05):
    """Check imbalance for continuous target.
    
    Args:
        series: Pandas series containing continuous target values
        skew_threshold: Threshold for skewness to consider imbalanced
        tail_threshold: Threshold for rare ratio in tail
        
    Returns:
        dict: Contains is_imbalanced, skewness, and rare_ratio
    """
    skewness = series.skew()

    upper_tail = series.quantile(0.95)
    rare_ratio = (series > upper_tail).mean()

    is_imbalanced = abs(skewness) > skew_threshold or rare_ratio < tail_threshold

    return {
        "is_imbalanced": is_imbalanced,
        "skewness": float(skewness),
        "rare_ratio": float(rare_ratio)
    }


def deep_flatten_and_convert(x):
    """Deep flatten and convert to native Python types.
    
    Handles nested arrays and converts numpy types to Python native types.
    
    Args:
        x: Value to flatten and convert
        
    Returns:
        Native Python type or None
    """
    while isinstance(x, (list, np.ndarray)):
        x = x[0] if len(x) > 0 else None

    if x is None:
        return None

    if isinstance(x, (np.integer, np.int64)):
        return int(x)

    if isinstance(x, (np.floating, np.float64)):
        if np.isnan(x):
            return None
        return int(round(x))

    if isinstance(x, (np.bool_,)):
        return int(x)

    if isinstance(x, bytes):
        return x.decode()

    return x


def clean_target_column(df, target_column):
    """Clean and normalize target column for binary classification.
    
    Args:
        df: DataFrame containing the target column
        target_column: Name of the target column
        
    Returns:
        tuple: (cleaned_df, metadata_dict)
    """
    before_rows = len(df)

    # Drop NaNs
    df = df.dropna(subset=[target_column])

    # Force numeric + normalize
    df[target_column] = (
        pd.to_numeric(df[target_column], errors="coerce")
          .round()
          .astype("Int64")
    )

    # Keep only valid binary
    df = df[df[target_column].isin([0, 1])]

    # Final hard cast
    df[target_column] = df[target_column].astype("int64")

    after_rows = len(df)

    return df, {"rows_dropped": before_rows - after_rows}


def find_target_column(dataset_id: str, table_name: str) -> str:
    """Find the target column from BigQuery table metadata.
    
    Searches for common target column names in the table schema.
    
    Args:
        dataset_id: BigQuery dataset ID (format: project.dataset)
        table_name: Name of the table
        
    Returns:
        str: Name of the target column, or None if not found
    """
    query = f"""
    SELECT column_name
    FROM `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_name}'
    """
    
    result = bq_client.query(query).to_dataframe()
    columns = result['column_name'].tolist()
    
    # Common target column names
    common_targets = ['target', 'label', 'class', 'y', 'output']
    for target in common_targets:
        if target in columns:
            return target
    
    # If no common target found, return the last column
    return columns[-1] if columns else None
