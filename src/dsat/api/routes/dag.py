"""DAG Generation API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

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
    table_name: str
    target_dataset: str
    target_column: Optional[str] = None
    transformations: List[Transformation]


@router.post("/generate_dag")
async def generate_dag(request: DAGRequest) -> Dict[str, Any]:
    """Generate Airflow DAG code from transformations."""
    try:
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            source_table=request.table_name,
            target_dataset=request.target_dataset
        )
        
        # Convert transformations to dict format
        transform_dicts = [
            {"column_name": t.column_name, "fe_method": t.fe_method}
            for t in request.transformations
        ]
        
        result = generator.generate(
            transformations=transform_dicts,
            target_column=request.target_column
        )
        
        return {
            "status": "success",
            "dag_id": result["dag_id"],
            "dag_code": result["dag_code"],
            "target_table": result["target_table_name"],
            "column_mapping": result["column_mapping"]
        }
        
    except Exception as e:
        logger.error(f"Error generating DAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview_sql")
async def preview_sql(request: DAGRequest) -> Dict[str, Any]:
    """Preview the SQL that would be generated."""
    try:
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            source_table=request.table_name,
            target_dataset=request.target_dataset
        )
        
        transform_dicts = [
            {"column_name": t.column_name, "fe_method": t.fe_method}
            for t in request.transformations
        ]
        
        sql = generator.preview_sql(
            transformations=transform_dicts,
            target_column=request.target_column
        )
        
        return {
            "status": "success",
            "sql": sql
        }
        
    except Exception as e:
        logger.error(f"Error previewing SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
