"""EDA API routes using KedroSession."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from dsat.api.kedro_runner import run_pipeline, get_kedro_session

logger = logging.getLogger(__name__)

router = APIRouter()


class TableConfig(BaseModel):
    """Request model for table configuration."""
    project_id: str
    dataset_id: str
    table_name: str
    sample_limit: int = 1500


class EDARequest(BaseModel):
    """Request model for EDA analysis."""
    project_id: str
    dataset_id: str
    table_name: str
    target_column: Optional[str] = None
    sample_limit: int = 1500


@router.get("/column_list")
async def column_list(
    project_id: str,
    dataset_id: str,
    table_name: str
) -> Dict[str, Any]:
    """Get list of columns from a BigQuery table."""
    try:
        from google.cloud import bigquery
        
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_name}"
        table = client.get_table(table_ref)
        
        columns = [
            {"name": field.name, "type": field.field_type}
            for field in table.schema
        ]
        
        return {
            "status": "success",
            "table": table_name,
            "columns": columns,
            "total_columns": len(columns)
        }
    except Exception as e:
        logger.error(f"Error getting column list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(request: EDARequest) -> Dict[str, Any]:
    """Run EDA analysis using Kedro pipeline with MLFlow tracking."""
    try:
        # Build runtime parameters
        extra_params = {
            "eda.table_config.project_id": request.project_id,
            "eda.table_config.dataset_id": request.dataset_id,
            "eda.table_config.table_name": request.table_name,
            "eda.sample_limit": request.sample_limit,
        }
        
        # Run EDA pipeline using KedroSession
        outputs = run_pipeline(
            pipeline_name="eda",
            extra_params=extra_params
        )
        
        # Get the EDA summary from outputs
        eda_summary = outputs.get("eda_summary", {})
        
        return {
            "status": "success",
            "data": eda_summary,
            "pipeline": "eda",
            "tracking": "mlflow"
        }
        
    except Exception as e:
        logger.error(f"Error in EDA analysis: {e}")
        # Fallback to direct node execution if Kedro session fails
        return await _analyze_direct(request)


async def _analyze_direct(request: EDARequest) -> Dict[str, Any]:
    """Fallback: Run EDA nodes directly without Kedro session."""
    try:
        from dsat.pipelines.eda.nodes import (
            load_data_from_bq,
            detect_column_types,
            compute_data_overview,
            compute_missing_values,
            compute_univariate_numerical,
            compute_univariate_categorical,
            generate_eda_summary,
        )
        
        df = load_data_from_bq(
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            table_name=request.table_name,
            sample_limit=request.sample_limit
        )
        
        column_types = detect_column_types(df)
        data_overview = compute_data_overview(df)
        missing_values = compute_missing_values(df)
        univariate_numerical = compute_univariate_numerical(df, column_types)
        univariate_categorical = compute_univariate_categorical(df, column_types)
        
        eda_summary = generate_eda_summary(
            data_overview=data_overview,
            missing_values=missing_values,
            univariate_numerical=univariate_numerical,
            univariate_categorical=univariate_categorical,
            column_types=column_types
        )
        
        return {
            "status": "success",
            "data": eda_summary,
            "pipeline": "direct",
            "tracking": "none"
        }
        
    except Exception as e:
        logger.error(f"Error in direct EDA analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eda_result")
async def eda_result(
    project_id: str,
    dataset_id: str,
    table_name: str
) -> Dict[str, Any]:
    """Get EDA result (runs analysis)."""
    request = EDARequest(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name
    )
    return await analyze(request)


@router.get("/pipelines")
async def list_pipelines() -> Dict[str, Any]:
    """List all available Kedro pipelines."""
    try:
        from dsat.api.kedro_runner import get_pipelines
        
        pipelines = get_pipelines()
        
        return {
            "status": "success",
            "pipelines": list(pipelines.keys())
        }
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))
