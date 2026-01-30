"""Feature Engineering Pipeline Nodes."""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def get_fe_recommendations(
    eda_summary: Dict[str, Any],
    target_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate feature engineering recommendations based on EDA.
    
    Args:
        eda_summary: EDA summary from EDA pipeline
        target_column: Optional target column name
    
    Returns:
        List of recommended transformations
    """
    recommendations = []
    
    column_types = eda_summary.get("Column Types", {})
    numerical_cols = column_types.get("numerical", [])
    categorical_cols = column_types.get("categorical", [])
    
    missing_data = eda_summary.get("Data Quality", {}).get("missing_values", {})
    missing_by_column = missing_data.get("missing_by_column", [])
    
    # Recommend imputation for columns with missing values
    for item in missing_by_column:
        col = item["column"]
        if col == target_column:
            continue
            
        if col in numerical_cols:
            recommendations.append({
                "column_name": col,
                "fe_method": "median_imputation",
                "reason": f"Column has {item['missing_pct']:.1f}% missing values"
            })
        elif col in categorical_cols:
            recommendations.append({
                "column_name": col,
                "fe_method": "mode_imputation",
                "reason": f"Column has {item['missing_pct']:.1f}% missing values"
            })
    
    # Recommend encoding for categorical columns
    for col in categorical_cols:
        if col == target_column:
            continue
        recommendations.append({
            "column_name": col,
            "fe_method": "label_encoding",
            "reason": "Categorical column needs encoding for ML"
        })
    
    # Recommend standardization for numerical columns
    for col in numerical_cols:
        if col == target_column:
            continue
        recommendations.append({
            "column_name": col,
            "fe_method": "standardization",
            "reason": "Numerical column - standardization for ML"
        })
    
    logger.info(f"Generated {len(recommendations)} FE recommendations")
    return recommendations


def generate_dag_code(
    recommendations: List[Dict[str, Any]],
    project_id: str,
    dataset_id: str,
    table_name: str,
    target_dataset: str,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """Generate Airflow DAG code from recommendations.
    
    Args:
        recommendations: List of FE recommendations
        project_id: GCP project ID
        dataset_id: Source dataset ID
        table_name: Source table name
        target_dataset: Target dataset for transformed data
        target_column: Optional target column
    
    Returns:
        Dict with DAG code and metadata
    """
    from dsat.common import DAGGenerator
    
    generator = DAGGenerator(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=table_name,
        target_dataset=target_dataset
    )
    
    # Convert recommendations to transformations
    transformations = [
        {"column_name": r["column_name"], "fe_method": r["fe_method"]}
        for r in recommendations
    ]
    
    result = generator.generate(transformations, target_column=target_column)
    
    logger.info(f"Generated DAG: {result['dag_id']}")
    return result


def preview_sql(dag_result: Dict[str, Any]) -> str:
    """Extract and return the SQL from the DAG.
    
    Args:
        dag_result: Result from generate_dag_code
    
    Returns:
        SQL query string
    """
    import re
    
    dag_code = dag_result.get("dag_code", "")
    match = re.search(r'"""(CREATE OR REPLACE.*?)"""', dag_code, re.DOTALL)
    
    return match.group(1).strip() if match else "SQL extraction failed"
