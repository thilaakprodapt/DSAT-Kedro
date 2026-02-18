"""EDA API routes using KedroSession.

Matches the original DataScienceAssistantTool API input/output format:
- POST /analyze: {dataset, table, chat?} -> {analysis, saved_to_bigquery, analysis_id, columns_count}
- GET /column_list: query params -> {project_id, dataset_id, table_name, columns}
- GET /eda_result: {dataset, table} -> {analysis, saved_to_bigquery, analysis_id, columns_count}
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from dsat.api.kedro_runner import run_pipeline, get_catalog

logger = logging.getLogger(__name__)

router = APIRouter()


class EDARequest(BaseModel):
    """Request model matching original DataScienceAssistantTool.
    
    Original: app/api/services/eda_service.py -> EDARequest
    """
    dataset: str
    table: str
    chat: Optional[str] = None


@router.get("/column_list")
async def column_list(
    project_id: str,
    dataset_id: str,
    table_name: str
) -> Dict[str, Any]:
    """Get list of columns from a BigQuery table.
    
    Matches original: app/api/routes/eda.py -> column_list
    Returns: {project_id, dataset_id, table_name, columns}
    """
    try:
        from google.cloud import bigquery
        
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_name}"
        table = client.get_table(table_ref)
        
        columns = [field.name for field in table.schema]
        
        return {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "table_name": table_name,
            "columns": columns
        }
    except Exception as e:
        logger.error(f"Error getting column list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(request: EDARequest) -> Dict[str, Any]:
    """Run EDA analysis using Kedro pipeline.
    
    Matches original: app/api/routes/eda.py -> analyze
    Input:  {dataset, table, chat?}
    Output: {analysis, saved_to_bigquery, analysis_id, columns_count}
    """
    try:
        # Build runtime parameters matching Kedro pipeline inputs
        # The dataset field contains the BigQuery dataset ID
        # We use the default project_id from parameters.yml
        extra_params = {
            "eda.table_config.dataset_id": request.dataset,
            "eda.table_config.table_name": request.table,
        }
        
        # Run EDA pipeline via KedroSession
        outputs = run_pipeline(
            pipeline_name="eda",
            extra_params=extra_params
        )
        
        # Get the EDA summary from outputs or catalog
        eda_summary = outputs.get("eda_summary")
        if not eda_summary:
            try:
                catalog = get_catalog()
                eda_summary = catalog.load("eda_summary")
            except Exception as e:
                logger.warning(f"Could not load eda_summary from catalog: {e}")
                eda_summary = {}
        
        # Get columns count
        column_types = outputs.get("column_types")
        if not column_types:
            try:
                catalog = get_catalog()
                column_types = catalog.load("column_types")
            except Exception:
                column_types = {}
        
        num_cols = len(column_types.get("numerical", []))
        cat_cols = len(column_types.get("categorical", []))
        columns_count = num_cols + cat_cols
        
        # Generate analysis_id matching original format
        analysis_id = f"{request.dataset}_{request.table}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Return in original format
        return {
            "analysis": eda_summary,
            "saved_to_bigquery": False,  # We save to Kedro catalog, not BQ directly
            "analysis_id": analysis_id,
            "columns_count": columns_count
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
            compute_outliers,
            compute_cardinality,
            compute_univariate_numerical,
            compute_univariate_categorical,
            find_target_column,
            generate_eda_summary,
        )
        
        # Use default project_id
        project_id = "cloud-practice-dev-2"
        
        df = load_data_from_bq(
            project_id=project_id,
            dataset_id=request.dataset,
            table_name=request.table,
            sample_limit=1500
        )
        
        column_types = detect_column_types(df)
        data_overview = compute_data_overview(df)
        missing_values = compute_missing_values(df)
        outliers = compute_outliers(df, column_types)
        cardinality = compute_cardinality(df)
        univariate_numerical = compute_univariate_numerical(df, column_types)
        univariate_categorical = compute_univariate_categorical(df, column_types)
        target_column = find_target_column(df, project_id)
        
        eda_summary = generate_eda_summary(
            df=df,
            data_overview=data_overview,
            missing_values=missing_values,
            outliers=outliers,
            cardinality=cardinality,
            univariate_numerical=univariate_numerical,
            univariate_categorical=univariate_categorical,
            column_types=column_types,
            target_column=target_column,
            project_id=project_id,
        )
        
        columns_count = len(column_types.get("numerical", [])) + len(column_types.get("categorical", []))
        analysis_id = f"{request.dataset}_{request.table}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "analysis": eda_summary,
            "saved_to_bigquery": False,
            "analysis_id": analysis_id,
            "columns_count": columns_count
        }
        
    except Exception as e:
        logger.error(f"Error in direct EDA analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eda_result")
async def eda_result(
    dataset: str,
    table: str
) -> Dict[str, Any]:
    """Get cached EDA result from Kedro catalog.
    
    Matches original: app/api/routes/eda.py -> eda_result
    Returns: {analysis, saved_to_bigquery, analysis_id, columns_count}
    """
    try:
        catalog = get_catalog()
        eda_summary = catalog.load("eda_summary")
        column_types = catalog.load("column_types")
        
        columns_count = len(column_types.get("numerical", [])) + len(column_types.get("categorical", []))
        analysis_id = f"{dataset}_{table}_cached"
        
        return {
            "analysis": eda_summary,
            "saved_to_bigquery": False,
            "analysis_id": analysis_id,
            "columns_count": columns_count
        }
    except Exception as e:
        logger.warning(f"No cached EDA result, running analysis: {e}")
        request = EDARequest(dataset=dataset, table=table)
        return await analyze(request)


@router.get("/pipelines")
async def list_pipelines() -> Dict[str, Any]:
    """List all available Kedro pipelines."""
    try:
        from dsat.api.kedro_runner import get_pipelines
        
        pipelines = get_pipelines()
        
        if isinstance(pipelines, dict):
            pipeline_names = list(pipelines.keys())
        else:
            pipeline_names = pipelines
        
        return {
            "status": "success",
            "pipelines": pipeline_names
        }
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))
