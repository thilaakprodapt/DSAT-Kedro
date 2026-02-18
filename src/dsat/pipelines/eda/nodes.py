"""EDA Pipeline Nodes.

Node functions for Exploratory Data Analysis pipeline.
Includes LLM-powered target detection and summary generation
matching the original DataScienceAssistantTool behavior.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from dsat.common.llm_utils import call_gemini, parse_llm_json

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
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "sample_data": df.head(5).to_dict(orient='records')
    }


def compute_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute missing value statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with missing value stats per column
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    missing_data = []
    for col in df.columns:
        missing_data.append({
            "column": col,
            "missing_count": int(missing_counts[col]),
            "missing_pct": float(missing_pct[col])
        })
    
    return {
        "total_missing_cells": int(missing_counts.sum()),
        "columns_with_missing": int((missing_counts > 0).sum()),
        "missing_values": missing_data
    }


def compute_outliers(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute outlier statistics for numerical columns using IQR method.
    
    Args:
        df: Input DataFrame
        column_types: Dict with 'numerical' and 'categorical' keys
    
    Returns:
        Dict with outlier stats per numerical column
    """
    numerical_cols = column_types.get("numerical", [])
    outlier_data = []
    
    for col in numerical_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outlier_count = int(((col_data < lower) | (col_data > upper)).sum())
        
        outlier_data.append({
            "column": col,
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(col_data) * 100, 2),
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        })
    
    return {"outliers": outlier_data}


def compute_cardinality(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute cardinality for each column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with cardinality info
    """
    cardinality_data = []
    for col in df.columns:
        unique = df[col].nunique()
        cardinality_data.append({
            "column": col,
            "unique_values": int(unique),
            "cardinality_ratio": round(unique / len(df), 4) if len(df) > 0 else 0,
            "is_high_cardinality": unique > 50
        })
    
    return {"Cardinality Check": cardinality_data}


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


def find_target_column(
    df: pd.DataFrame,
    project_id: str,
) -> str:
    """Auto-detect the most likely target column using Gemini.
    
    Sends column metadata (dtype, unique values, sample values) to Gemini
    and asks it to identify the most likely target column.
    
    Args:
        df: Input DataFrame
        project_id: GCP project ID (for Vertex AI init)
    
    Returns:
        Name of the detected target column
    """
    # Build column metadata
    metadata = {}
    for col in df.columns:
        metadata[col] = {
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().astype(str).unique()[:5].tolist()
        }

    prompt = f"""You are an expert data scientist.
Your task: Identify the most likely target column from the dataset.
Dataset columns:
    {df.columns.tolist()}
Column metadata:
    {json.dumps(metadata)}
Rule: Identify the MOST LIKELY target column
Return only the column_name"""

    response_text = call_gemini(prompt, project_id=project_id)
    target = response_text.strip().replace(" ", "_")
    
    # Validate the target exists in columns
    if target not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == target.lower():
                target = col
                break
        else:
            logger.warning(f"LLM returned target '{target}' not in columns, using first column")
            target = df.columns[0]
    
    logger.info(f"Target column detected: {target}")
    return target


def generate_eda_summary(
    df: pd.DataFrame,
    data_overview: Dict[str, Any],
    missing_values: Dict[str, Any],
    outliers: Dict[str, Any],
    cardinality: Dict[str, Any],
    univariate_numerical: List[Dict[str, Any]],
    univariate_categorical: List[Dict[str, Any]],
    column_types: Dict[str, List[str]],
    target_column: str,
    project_id: str,
) -> Dict[str, Any]:
    """Generate complete EDA summary using Gemini.
    
    Sends all computed statistics to Gemini and asks it to generate a
    structured JSON report with detailed summaries for each section.
    Matches the original DataScienceAssistantTool eda_service.py output format.
    
    Args:
        df: Input DataFrame
        data_overview: Basic data stats
        missing_values: Missing value analysis
        outliers: Outlier analysis
        cardinality: Cardinality analysis
        univariate_numerical: Numerical column stats
        univariate_categorical: Categorical column stats
        column_types: Column type classification
        target_column: Detected target column name
        project_id: GCP project ID (for Vertex AI init)
    
    Returns:
        Complete EDA summary dict (structured by Gemini)
    """
    num_cols = column_types.get("numerical", [])
    cat_cols = column_types.get("categorical", [])

    # Build the prompt matching original eda_service.py
    prompt = f"""
        You are an expert data scientist.
        I have computed the following statistics for the dataset:

        Data Overview: {json.dumps(data_overview)}
        Data Quality - Missing Values: {json.dumps(missing_values)}
        Data Quality - Outliers: {json.dumps(outliers)}
        Data Quality - Cardinality: {json.dumps(cardinality)}
        Univariate Numerical: {json.dumps(univariate_numerical)}
        Univariate Categorical: {json.dumps(univariate_categorical)}
        Target Column: {target_column}

        Your task is to generate a JSON response with the EXACT structure provided below.
        Fill in the values using the statistics provided above.
        Write detailed summaries for the analysis sections based on the stats.

        Output JSON Structure:
        {{
         "Data Overview": {{
             "shape": {{"rows": <number>, "columns": <number>}},
             "feature_types": {{
                 "numerical": {json.dumps(num_cols)},
                 "categorical": {json.dumps(cat_cols)},
                 "boolean": [],
                 "datetime": [],
                 "text": [],
                 "high_cardinality": []
             }},
             "sample_data": {json.dumps(data_overview.get("sample_data", []))}
         }},
                    "Data quality":{{
         "missing_values": {json.dumps(missing_values.get("missing_values", []))},
                    "outliers": {json.dumps(outliers.get("outliers", []))},
                    "Cardinality Check": {json.dumps(cardinality.get("Cardinality Check", []))}
                    }},
         "Univariate Analysis": {{
             "numerical": {json.dumps(univariate_numerical)},
             "summary": "Give a detailed summary of the numerical analysis",
             "categorical": {json.dumps(univariate_categorical)},
             "summary_cat": "Give a detailed summary of the categorical analysis"
         }},
    
         "Bivariate Analysis": {{
             "Target_column":"Based on column metadata and dataset patterns, {target_column} has been selected as the target column.",
             "Numerical vs Target": {{
             "summary": "<detailed summary of relationships with target>"
             }},
             "Categorical vs Target": {{
             "summary": "<detailed summary of categorical features with target>"
             }}
         }},
         "Summary": {{
             "summary":"From the whole analysis give a comprehensive summary on the data, analysis and insights"
         }}
         }}

        IMPORTANT INSTRUCTIONS:
        1. Make sure the output JSON follows the structure exactly as shown
        2. Return ONLY valid JSON - no markdown, no code blocks, no preamble
        3. Ensure all numbers are actual numbers (not strings)
        4. All arrays must contain objects as shown in the structure
        5. Provide comprehensive summaries for all visualization sections
"""

    response_text = call_gemini(prompt, project_id=project_id)
    logger.info(f"Gemini EDA response: {response_text[:200]}...")

    # Parse the LLM response
    data = parse_llm_json(response_text)

    if data:
        logger.info("Successfully parsed Gemini EDA summary")
        # Ensure target_column is preserved in the response
        if "Bivariate Analysis" in data:
            data["Bivariate Analysis"]["Target_column"] = (
                f"Based on column metadata and dataset patterns, "
                f"{target_column} has been selected as the target column."
            )
        return data
    else:
        # Fallback: return raw stats without LLM summaries
        logger.warning("Gemini parsing failed, returning raw stats fallback")
        return {
            "Data Overview": {
                "shape": data_overview["shape"],
                "feature_types": {
                    "numerical": num_cols,
                    "categorical": cat_cols,
                    "boolean": [],
                    "datetime": [],
                    "text": [],
                    "high_cardinality": [],
                },
                "sample_data": data_overview.get("sample_data", []),
            },
            "Data quality": {
                "missing_values": missing_values.get("missing_values", []),
                "outliers": outliers.get("outliers", []),
                "Cardinality Check": cardinality.get("Cardinality Check", []),
            },
            "Univariate Analysis": {
                "numerical": univariate_numerical,
                "categorical": univariate_categorical,
            },
            "Bivariate Analysis": {
                "Target_column": f"Based on column metadata and dataset patterns, {target_column} has been selected as the target column.",
            },
            "Summary": {
                "summary": f"Dataset has {data_overview['shape']['rows']} rows and {data_overview['shape']['columns']} columns. "
                           f"Found {len(num_cols)} numerical and {len(cat_cols)} categorical features. "
                           f"{missing_values['columns_with_missing']} columns have missing values."
            },
        }
