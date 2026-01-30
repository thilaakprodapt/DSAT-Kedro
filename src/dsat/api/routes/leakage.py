"""Leakage Detection API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class LeakageRequest(BaseModel):
    """Request model for leakage detection."""
    project_id: str
    dataset_id: str
    table_name: str
    target_column: str
    feature_columns: Optional[List[str]] = None


@router.post("/detect")
async def detect_leakage(request: LeakageRequest) -> Dict[str, Any]:
    """Detect potential data leakage in features."""
    try:
        from google.cloud import bigquery
        import pandas as pd
        
        client = bigquery.Client(project=request.project_id)
        
        # Load data
        query = f"""
            SELECT * 
            FROM `{request.project_id}.{request.dataset_id}.{request.table_name}` 
            LIMIT 5000
        """
        df = client.query(query).to_dataframe()
        
        # Get feature columns
        if request.feature_columns:
            feature_cols = request.feature_columns
        else:
            feature_cols = [c for c in df.columns if c != request.target_column]
        
        # Check for leakage indicators
        warnings = []
        
        # 1. Perfect or near-perfect correlation with target
        if request.target_column in df.columns:
            target = df[request.target_column]
            
            for col in feature_cols:
                if col not in df.columns:
                    continue
                    
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        corr = df[col].corr(target.astype(float))
                        if abs(corr) > 0.95:
                            warnings.append({
                                "column": col,
                                "type": "high_correlation",
                                "severity": "high",
                                "value": round(corr, 4),
                                "message": f"Suspiciously high correlation ({corr:.2f}) with target"
                            })
                except Exception:
                    pass
        
        # 2. Check for future-looking column names
        future_keywords = ['future', 'next', 'will', 'outcome', 'result', 'target', 'label']
        for col in feature_cols:
            col_lower = col.lower()
            for keyword in future_keywords:
                if keyword in col_lower and col != request.target_column:
                    warnings.append({
                        "column": col,
                        "type": "suspicious_name",
                        "severity": "medium",
                        "message": f"Column name contains '{keyword}' - may indicate leakage"
                    })
                    break
        
        # 3. Check for columns that are identical to target
        if request.target_column in df.columns:
            for col in feature_cols:
                if col in df.columns and col != request.target_column:
                    if df[col].equals(df[request.target_column]):
                        warnings.append({
                            "column": col,
                            "type": "identical_to_target",
                            "severity": "critical",
                            "message": "Column is identical to target - definite leakage!"
                        })
        
        return {
            "status": "success",
            "target_column": request.target_column,
            "features_analyzed": len(feature_cols),
            "leakage_warnings": warnings,
            "total_warnings": len(warnings),
            "risk_level": "high" if any(w["severity"] == "critical" for w in warnings) 
                         else "medium" if warnings else "low"
        }
        
    except Exception as e:
        logger.error(f"Error detecting leakage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
