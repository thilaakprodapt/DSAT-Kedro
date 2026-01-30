"""DAG Generation API routes - matching original endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


class Transformation(BaseModel):
    """A single transformation specification."""
    column_name: str
    fe_method: str


class DAGRequest(BaseModel):
    """Request model for DAG generation."""
    project_id: str
    dataset_id: str
    source_table: str
    target_dataset: str
    target_column: Optional[str] = None
    transformation: List[Transformation]


class SaveDAGRequest(BaseModel):
    """Request model for saving DAG."""
    dag_code: str
    dag_name: str
    target_table_name: str
    target_dataset: str
    source_table: str
    input_data: dict


@router.post("/generate_dag")
async def generate_dag(input: dict) -> Dict[str, Any]:
    """Generate Airflow DAG code from transformations.
    
    Matches original endpoint: POST /Transformation/generate_dag
    """
    try:
        from dsat.common import DAGGenerator
        
        project_id = input.get("project_id")
        dataset_id = input.get("dataset_id")
        source_table = input.get("source_table")
        target_dataset = input.get("target_dataset")
        transformations = input.get("transformation", [])
        target_column = input.get("target_column")
        
        if not all([project_id, dataset_id, source_table, target_dataset, transformations]):
            raise ValueError("Missing required fields")
        
        generator = DAGGenerator(
            project_id=project_id,
            dataset_id=dataset_id,
            source_table=source_table,
            target_dataset=target_dataset
        )
        
        result = generator.generate(
            transformations=transformations,
            target_column=target_column
        )
        
        result["input_data"] = input
        result["generation_method"] = "template"
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating DAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save_dag")
def save_dag(request: SaveDAGRequest) -> Dict[str, Any]:
    """Save DAG to file system and trigger it.
    
    Matches original endpoint: POST /Transformation/save_dag
    """
    try:
        dag_name = request.dag_name
        dag_code = request.dag_code
        
        # Save DAG file (adjust path for your environment)
        # On GCP VM: /home/airflow/dags/
        # Locally: ./dags/
        dag_dir = os.environ.get("DAG_FOLDER", "./dags")
        os.makedirs(dag_dir, exist_ok=True)
        
        file_path = os.path.join(dag_dir, f"{dag_name}.py")
        with open(file_path, "w") as f:
            f.write(dag_code)
        
        logger.info(f"DAG saved at {file_path}")
        
        return {
            "status": "success",
            "dag_name": dag_name,
            "target_table_name": request.target_table_name,
            "target_dataset": request.target_dataset,
            "file_path": file_path,
            "message": f"DAG saved successfully at {file_path}"
        }
        
    except Exception as e:
        logger.error(f"Error saving DAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview_sql")
async def preview_sql(input: dict) -> Dict[str, Any]:
    """Preview the SQL that would be generated."""
    try:
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id=input.get("project_id"),
            dataset_id=input.get("dataset_id"),
            source_table=input.get("source_table"),
            target_dataset=input.get("target_dataset")
        )
        
        transformations = input.get("transformation", [])
        target_column = input.get("target_column")
        
        sql = generator.preview_sql(
            transformations=transformations,
            target_column=target_column
        )
        
        return {
            "status": "success",
            "sql": sql
        }
        
    except Exception as e:
        logger.error(f"Error previewing SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fe_chat_check")
def fe_chat_check(input: str) -> Dict[str, Any]:
    """Check if input is a valid FE technique.
    
    Matches original endpoint: POST /Transformation/fe_chat_check
    """
    try:
        # Simple validation without AI for now
        valid_techniques = {
            "standardization", "normalization", "log_transformation",
            "label_encoding", "frequency_encoding", "target_encoding",
            "one-hot encoding", "one_hot_encoding", "ordinal_encoding",
            "min-max scaling", "robust_scaling", "pca", "svd",
            "mean_imputation", "median_imputation", "mode_imputation",
            "impute_mean", "impute_median", "impute_mode",
            "binning", "clip_outliers", "winsorize", "smote"
        }
        
        input_lower = input.lower().strip()
        is_valid = input_lower in valid_techniques
        
        suggestion = ""
        if not is_valid:
            if "encod" in input_lower:
                suggestion = "Try: one-hot encoding, label encoding, target encoding, ordinal encoding, frequency encoding"
            elif "scal" in input_lower:
                suggestion = "Try: standardization, min-max scaling, robust scaling, normalization"
            elif "transform" in input_lower:
                suggestion = "Try: log transformation, standardization, normalization"
            elif "imput" in input_lower or "missing" in input_lower:
                suggestion = "Try: mean_imputation, median_imputation, mode_imputation"
            else:
                suggestion = "Check that the input is a valid ML feature engineering technique"
        
        return {
            "valid": is_valid,
            "suggestion": suggestion
        }
        
    except Exception as e:
        logger.error(f"Error in fe_chat_check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available_transformations")
async def available_transformations() -> Dict[str, Any]:
    """Get list of available transformations."""
    return {
        "status": "success",
        "categories": {
            "numerical": [
                "standardization", "normalization", "log_transformation",
                "sqrt_transformation", "binning", "robust_scaling"
            ],
            "categorical": [
                "label_encoding", "frequency_encoding", "target_encoding"
            ],
            "missing_values": [
                "impute_mean", "impute_median", "impute_mode",
                "impute_constant", "missing_indicator"
            ],
            "outliers": [
                "clip_outliers", "clip_iqr", "winsorize"
            ]
        }
    }
