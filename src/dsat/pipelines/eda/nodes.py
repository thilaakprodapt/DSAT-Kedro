"""EDA Pipeline Nodes.

Node functions for Exploratory Data Analysis pipeline.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def load_data_from_bq(
    project_id: str,
    dataset_id: str,
    table_name: str,
    sample_limit: int = 1500
) -> pd.DataFrame:
    """Load data from BigQuery table.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: BigQuery table name
        sample_limit: Maximum rows to sample
    
    Returns:
        DataFrame with sampled data
    """
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT * 
        FROM `{project_id}.{dataset_id}.{table_name}` 
        LIMIT {sample_limit}
    """
    
    logger.info(f"Loading data from {project_id}.{dataset_id}.{table_name}")
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect numerical and categorical columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with 'numerical' and 'categorical' column lists
    """
    numerical = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    logger.info(f"Detected {len(numerical)} numerical, {len(categorical)} categorical columns")
    
    return {
        "numerical": numerical,
        "categorical": categorical
    }


def compute_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic data overview statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with overview stats
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "sample_data": df.head(5).to_dict(orient='records')
    }


def compute_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute missing value statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with missing value stats
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    missing_data = []
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_data.append({
                "column": col,
                "missing_count": int(missing_counts[col]),
                "missing_pct": float(missing_pct[col])
            })
    
    return {
        "total_missing_cells": int(missing_counts.sum()),
        "columns_with_missing": len(missing_data),
        "missing_by_column": missing_data
    }


def compute_univariate_numerical(
    df: pd.DataFrame, 
    column_types: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Compute univariate statistics for numerical columns.
    
    Args:
        df: Input DataFrame
        column_types: Dict with 'numerical' and 'categorical' keys
    
    Returns:
        List of stats for each numerical column
    """
    numerical_cols = column_types.get("numerical", [])
    stats = []
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        stats.append({
            "column_name": col,
            "count": int(len(col_data)),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "q25": float(col_data.quantile(0.25)),
            "q75": float(col_data.quantile(0.75)),
            "skewness": float(col_data.skew()) if len(col_data) > 2 else 0,
            "kurtosis": float(col_data.kurtosis()) if len(col_data) > 3 else 0,
        })
    
    return stats


def compute_univariate_categorical(
    df: pd.DataFrame, 
    column_types: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Compute univariate statistics for categorical columns.
    
    Args:
        df: Input DataFrame
        column_types: Dict with 'numerical' and 'categorical' keys
    
    Returns:
        List of stats for each categorical column
    """
    categorical_cols = column_types.get("categorical", [])
    stats = []
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        top_5 = value_counts.head(5)
        
        stats.append({
            "column_name": col,
            "unique_count": int(df[col].nunique()),
            "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "mode_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "top_5_frequencies": [
                {"value": str(v), "count": int(c)} 
                for v, c in zip(top_5.index, top_5.values)
            ]
        })
    
    return stats


def generate_eda_summary(
    data_overview: Dict[str, Any],
    missing_values: Dict[str, Any],
    univariate_numerical: List[Dict[str, Any]],
    univariate_categorical: List[Dict[str, Any]],
    column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Generate complete EDA summary.
    
    Args:
        data_overview: Basic data stats
        missing_values: Missing value analysis
        univariate_numerical: Numerical column stats
        univariate_categorical: Categorical column stats
        column_types: Column type classification
    
    Returns:
        Complete EDA summary dict
    """
    return {
        "Data Overview": {
            "shape": {
                "rows": data_overview["rows"],
                "columns": data_overview["columns"]
            },
            "columns": data_overview["column_names"],
            "dtypes": data_overview["dtypes"],
            "sample_data": data_overview["sample_data"]
        },
        "Data Quality": {
            "missing_values": missing_values
        },
        "Column Types": column_types,
        "Univariate Analysis": {
            "numerical": univariate_numerical,
            "categorical": univariate_categorical
        },
        "Summary": {
            "total_rows": data_overview["rows"],
            "total_columns": data_overview["columns"],
            "numerical_columns": len(column_types["numerical"]),
            "categorical_columns": len(column_types["categorical"]),
            "columns_with_missing": missing_values["columns_with_missing"]
        }
    }
