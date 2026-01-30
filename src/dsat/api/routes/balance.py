"""Data Balancing API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class BalanceRequest(BaseModel):
    """Request model for data balancing."""
    project_id: str
    dataset_id: str
    table_name: str
    target_column: str
    method: str = "smote"  # smote, oversample, undersample


@router.post("/balance_data")
async def balance_data(request: BalanceRequest) -> Dict[str, Any]:
    """Balance imbalanced dataset.
    
    Matches original endpoint: POST /DataBalancing/balance_data
    """
    try:
        from google.cloud import bigquery
        import pandas as pd
        
        client = bigquery.Client(project=request.project_id)
        
        # Load data
        query = f"""
            SELECT * 
            FROM `{request.project_id}.{request.dataset_id}.{request.table_name}`
        """
        df = client.query(query).to_dataframe()
        
        # Check class distribution
        target = request.target_column
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in table")
        
        class_counts = df[target].value_counts().to_dict()
        total_samples = len(df)
        
        # Calculate imbalance ratio
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        # Recommendations based on imbalance
        if imbalance_ratio < 1.5:
            recommendation = "Dataset is relatively balanced. No balancing needed."
            severity = "low"
        elif imbalance_ratio < 3:
            recommendation = "Moderate imbalance. Consider using class weights or SMOTE."
            severity = "medium"
        else:
            recommendation = "Severe imbalance. Recommend SMOTE or undersampling."
            severity = "high"
        
        return {
            "status": "success",
            "target_column": target,
            "class_distribution": class_counts,
            "total_samples": total_samples,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "severity": severity,
            "recommendation": recommendation,
            "method_requested": request.method,
            "message": f"Analyzed class distribution for {target}"
        }
        
    except Exception as e:
        logger.error(f"Error in data balancing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
