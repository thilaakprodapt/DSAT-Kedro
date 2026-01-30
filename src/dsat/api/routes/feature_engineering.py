"""Feature Engineering API routes."""

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


class FERecommendation(BaseModel):
    """A single FE recommendation."""
    column_name: str
    fe_method: str
    reason: Optional[str] = None


@router.post("/recommendations")
async def get_recommendations(request: FERequest) -> Dict[str, Any]:
    """Get feature engineering recommendations based on EDA."""
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
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error getting FE recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def apply_transformations(
    request: FERequest,
    transformations: List[FERecommendation]
) -> Dict[str, Any]:
    """Apply feature engineering transformations."""
    try:
        from dsat.common import SQLTemplateEngine
        
        engine = SQLTemplateEngine(
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            source_table=request.table_name
        )
        
        # Convert to dict format
        transform_dicts = [
            {"column_name": t.column_name, "fe_method": t.fe_method}
            for t in transformations
        ]
        
        sql = engine.render_select_statement(transform_dicts, include_original=True)
        
        return {
            "status": "success",
            "preview_sql": sql,
            "transformations_count": len(transformations)
        }
        
    except Exception as e:
        logger.error(f"Error applying transformations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
