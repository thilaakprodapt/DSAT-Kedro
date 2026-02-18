"""Feature Engineering API routes.

Matches the original DataScienceAssistantTool API input/output format:
- POST /FE_analyze: {eda_output, target_column, chat?} -> {feature_engineering, imbalance_analysis, bias_detection, balancing_recommendations}
- GET /fe_result: {eda_output, target_column} -> same as above
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from dsat.api.kedro_runner import run_pipeline, get_catalog

logger = logging.getLogger(__name__)

router = APIRouter()


class FERequest(BaseModel):
    """Request model matching original DataScienceAssistantTool.

    Original: app/api/services/feature_engg_service.py -> FERequest
    """
    eda_output: dict  # Full EDA output from /analyze or /eda_result
    target_column: str
    chat: Optional[str] = None


@router.post("/FE_analyze")
async def fe_analyze(request: FERequest) -> Dict[str, Any]:
    """Run Feature Engineering analysis using Kedro pipeline.

    Matches original: app/api/routes/feature_engg.py -> fe_analyze
    Input:  {eda_output, target_column, chat?}
    Output: {feature_engineering, imbalance_analysis, bias_detection, balancing_recommendations}
    """
    try:
        # Extract EDA analysis data from the input
        eda_analysis = request.eda_output.get("analysis", request.eda_output)

        # Try to run via Kedro pipeline
        # First, we need eda_summary to be available in the catalog
        # The EDA output from /analyze is the eda_summary
        try:
            catalog = get_catalog()

            # Save the EDA summary to catalog so FE pipeline can use it
            if "eda_summary" in catalog.list():
                catalog.save("eda_summary", eda_analysis)

            # Run FE pipeline with target_column override
            extra_params = {
                "fe.target_column": request.target_column,
            }

            outputs = run_pipeline(
                pipeline_name="fe",
                extra_params=extra_params
            )

            # Get FE recommendations from outputs or catalog
            fe_recommendations = outputs.get("fe_recommendations")
            if not fe_recommendations:
                fe_recommendations = catalog.load("fe_recommendations")

        except Exception as kedro_error:
            logger.warning(f"Kedro pipeline failed, using direct execution: {kedro_error}")
            fe_recommendations = await _fe_analyze_direct(eda_analysis, request.target_column)

        # Return in original format (the 4 top-level sections)
        return fe_recommendations

    except Exception as e:
        logger.error(f"Error in FE analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _fe_analyze_direct(eda_analysis: dict, target_column: str) -> Dict[str, Any]:
    """Fallback: Run FE node directly without Kedro session."""
    from dsat.pipelines.feature_engineering.nodes import get_fe_recommendations

    project_id = "cloud-practice-dev-2"

    return get_fe_recommendations(
        eda_summary=eda_analysis,
        target_column=target_column,
        project_id=project_id,
    )


@router.get("/fe_result")
async def get_fe_result(
    eda_output: Optional[str] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Get cached FE result from Kedro catalog.

    Matches original: app/api/routes/feature_engg.py -> get_fe_result
    """
    try:
        catalog = get_catalog()
        fe_recommendations = catalog.load("fe_recommendations")
        return fe_recommendations
    except Exception as e:
        logger.warning(f"No cached FE result: {e}")
        raise HTTPException(
            status_code=404,
            detail="No cached FE result. Run POST /FE_analyze first."
        )
