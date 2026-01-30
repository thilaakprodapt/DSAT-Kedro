"""Feature Engineering API routes - matching original endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class FERequest(BaseModel):
    """Request model for Feature Engineering."""
    project_id: str
    dataset_id: str
    table_name: str
    target_column: Optional[str] = None


@router.post("/FE_analyze")
async def fe_analyze(request: FERequest) -> Dict[str, Any]:
    """Run Feature Engineering analysis.
    
    Matches original endpoint: POST /Feature Engineering/FE_analyze
    """
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
        from dsat.pipelines.feature_engineering.nodes import get_fe_recommendations
        
        # Run EDA first
        df = load_data_from_bq(
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            table_name=request.table_name,
            sample_limit=1500
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
        
        # Get FE recommendations
        recommendations = get_fe_recommendations(
            eda_summary=eda_summary,
            target_column=request.target_column
        )
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "column_types": column_types,
            "eda_summary": eda_summary
        }
        
    except Exception as e:
        logger.error(f"Error in FE analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fe_result")
async def get_fe_result(
    project_id: str,
    dataset_id: str,
    table_name: str,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """Get cached FE result (runs analysis if not cached).
    
    Matches original endpoint: GET /Feature Engineering/fe_result
    """
    request = FERequest(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
        target_column=target_column
    )
    return await fe_analyze(request)


@router.post("/recommendations")
async def get_recommendations(request: FERequest) -> Dict[str, Any]:
    """Alternative endpoint for FE recommendations."""
    return await fe_analyze(request)
