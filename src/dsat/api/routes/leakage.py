"""Leakage Detection API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dsat.common.leakage_utils import (
    set_phase_status,
    save_leakage_to_bq
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize BigQuery client with service account
def _get_credentials():
    """Get credentials from service account file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        return service_account.Credentials.from_service_account_file(credentials_path)
    return None

credentials = _get_credentials()
bq_client = bigquery.Client(credentials=credentials) if credentials else bigquery.Client()
project_id = os.getenv("GCP_PROJECT_ID", bq_client.project)



class LeakageRequest(BaseModel):
    project_id: str
    dataset_id: str
    table_name: str
    target_column: str



class LeakageMitigationRequest(BaseModel):
    """Request model for leakage mitigation."""
    dataset_id: str
    table_name: str
    columns_to_drop: List[str]


# @router.post("/detect")
# async def detect_leakage(request: LeakageRequest) -> Dict[str, Any]:
#     """Detect potential data leakage in features."""
#     try:
#         from google.cloud import bigquery
#         import pandas as pd
        
#         client = bigquery.Client(project=request.project_id)
        
#         # Load data
#         query = f"""
#             SELECT * 
#             FROM `{request.project_id}.{request.dataset_id}.{request.table_name}` 
#             LIMIT 5000
#         """
#         df = client.query(query).to_dataframe()
        
#         # Get feature columns
#         if request.feature_columns:
#             feature_cols = request.feature_columns
#         else:
#             feature_cols = [c for c in df.columns if c != request.target_column]
        
#         # Check for leakage indicators
#         warnings = []
        
#         # 1. Perfect or near-perfect correlation with target
#         if request.target_column in df.columns:
#             target = df[request.target_column]
            
#             for col in feature_cols:
#                 if col not in df.columns:
#                     continue
                    
#                 try:
#                     if df[col].dtype in ['int64', 'float64']:
#                         corr = df[col].corr(target.astype(float))
#                         if abs(corr) > 0.95:
#                             warnings.append({
#                                 "column": col,
#                                 "type": "high_correlation",
#                                 "severity": "high",
#                                 "value": round(corr, 4),
#                                 "message": f"Suspiciously high correlation ({corr:.2f}) with target"
#                             })
#                 except Exception:
#                     pass
        
#         # 2. Check for future-looking column names
#         future_keywords = ['future', 'next', 'will', 'outcome', 'result', 'target', 'label']
#         for col in feature_cols:
#             col_lower = col.lower()
#             for keyword in future_keywords:
#                 if keyword in col_lower and col != request.target_column:
#                     warnings.append({
#                         "column": col,
#                         "type": "suspicious_name",
#                         "severity": "medium",
#                         "message": f"Column name contains '{keyword}' - may indicate leakage"
#                     })
#                     break
        
#         # 3. Check for columns that are identical to target
#         if request.target_column in df.columns:
#             for col in feature_cols:
#                 if col in df.columns and col != request.target_column:
#                     if df[col].equals(df[request.target_column]):
#                         warnings.append({
#                             "column": col,
#                             "type": "identical_to_target",
#                             "severity": "critical",
#                             "message": "Column is identical to target - definite leakage!"
#                         })
        
#         return {
#             "status": "success",
#             "target_column": request.target_column,
#             "features_analyzed": len(feature_cols),
#             "leakage_warnings": warnings,
#             "total_warnings": len(warnings),
#             "risk_level": "high" if any(w["severity"] == "critical" for w in warnings) 
#                          else "medium" if warnings else "low"
#         }
        
#     except Exception as e:
#         logger.error(f"Error detecting leakage: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/leakage_detection", tags=["Leakage Detection"])
async def leakage_detection(request: LeakageRequest) -> Dict[str, Any]:
    try:
        import json
        import re
        from google.cloud import bigquery
        from vertexai.generative_models import GenerativeModel

        client = bigquery.Client(project=request.project_id)

        table = request.table_name.replace("_transformed", "")

        # ---------------------------------------------------
        # 1. Mark Status: IN PROGRESS
        # ---------------------------------------------------
        set_phase_status(client, table, "Leakage_status", "inprogress")

        full_table_id = f"{request.project_id}.{request.dataset_id}.{request.table_name}"
        query = f"SELECT * FROM `{full_table_id}`"
        df = client.query(query).to_dataframe()

        # Keep prompt small
        sample_json = df.head(200).to_json(orient="records")

        # ---------------------------------------------------
        # 2. LLM Prompt
        # ---------------------------------------------------
        model = GenerativeModel("gemini-2.0-flash")

        prompt = f"""
You are a machine learning data leakage detection expert.

Target column: {request.target_column}

Dataset sample (max 200 rows):
{sample_json}

Return STRICT JSON only in the following format:

{{
  "summary": {{
    "high_risk": <number>,
    "medium_risk": <number>,
    "low_risk": <number>
  }},
  "high_risk_features": [
    {{
      "feature": "<column_name>",
      "reason": "<reason>"
    }}
  ],
  "medium_risk_features": [
    {{
      "feature": "<column_name>",
      "reason": "<reason>"
    }}
  ],
  "low_risk_features": [
    {{
      "feature": "<column_name>",
      "reason": "<reason>"
    }}
  ]
}}

Rules:
- Output valid JSON only
- No explanations outside JSON
- Features must exist in dataset
- High-risk = future info or direct target leakage
- Medium-risk = strong proxy or suspicious correlation
- Low-risk = weak or needs review
"""

        response = model.generate_content(prompt)
        response_text = response.text

        # ---------------------------------------------------
        # 3. Extract JSON
        # ---------------------------------------------------
        json_match = re.search(
            r'```(?:json)?\s*([\s\S]*?)\s*```',
            response_text,
            re.DOTALL
        )

        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if not json_str:
            set_phase_status(client, table, "Leakage_status", "failed")
            return {"status": "error", "message": "No valid JSON returned from LLM"}

        # ---------------------------------------------------
        # 4. Parse JSON
        # ---------------------------------------------------
        try:
            leakage_data = json.loads(json_str)
        except json.JSONDecodeError:
            set_phase_status(client, table, "Leakage_status", "failed")
            return {
                "status": "error",
                "message": "JSON parsing failed",
                "raw_output": response_text
            }

        # ---------------------------------------------------
        # 5. Save to BigQuery
        # ---------------------------------------------------
        _id = save_leakage_to_bq(
            client,
            request.dataset_id,
            table,
            leakage_data,
            json_str
        )

        # ---------------------------------------------------
        # 6. Mark Completed
        # ---------------------------------------------------
        # set_phase_status(client, table, "Leakage_status", "completed")

        return {
            "status": "success",
            "id": _id,
            "leakage_detection": leakage_data
        }

    except Exception as e:
        set_phase_status(client, table, "Leakage_status", "failed")
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/leakage_mitigation")
def leakage_mitigation(request: LeakageMitigationRequest):
    """
    Mitigate data leakage by dropping specified columns.
    
    Performs case-insensitive column matching, protects target column,
    and saves cleaned data to BigQuery.
    """
    try:
        from dsat.common.imbalance_utils import find_target_column
        
        dataset_id = request.dataset_id
        table_name = request.table_name
        user_columns_to_drop = request.columns_to_drop
        
        # --------------------------------------------------
        # 1️⃣ Load table from BigQuery
        # --------------------------------------------------
        query = f"SELECT * FROM `{dataset_id}.{table_name}`"
        df = bq_client.query(query).to_dataframe()

        if df.empty:
            return {
                "status": "error",
                "message": "Input table is empty"
            }

        # --------------------------------------------------
        # 2️⃣ Detect target column (existing logic)
        # --------------------------------------------------
        target_column = find_target_column(dataset_id, table_name)

        # --------------------------------------------------
        # 3️⃣ Case-insensitive column normalization
        # --------------------------------------------------
        column_name_map = {col.lower(): col for col in df.columns}

        safe_columns_to_drop = []

        for user_col in user_columns_to_drop:
            user_col_clean = user_col.strip().lower()

            # Never drop target column
            if user_col_clean == target_column.lower():
                continue

            # Drop only if column exists
            if user_col_clean in column_name_map:
                safe_columns_to_drop.append(column_name_map[user_col_clean])

        # --------------------------------------------------
        # 4️⃣ Drop columns
        # --------------------------------------------------
        df_clean = df.drop(columns=safe_columns_to_drop)

        # --------------------------------------------------
        # 5️⃣ Save back to BigQuery
        # --------------------------------------------------
        output_dataset = "DS_Leakage_Dataset"
        output_table = f"{table_name}_dropped"
        table_id = f"{project_id}.{output_dataset}.{output_table}"

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        )

        load_job = bq_client.load_table_from_dataframe(
            df_clean,
            table_id,
            job_config=job_config
        )
        load_job.result()

        # --------------------------------------------------
        # 6️⃣ Audit info
        # --------------------------------------------------
        ignored_columns = [
            col for col in user_columns_to_drop
            if col.strip().lower() not in column_name_map
            or col.strip().lower() == target_column.lower()
        ]

        return {
            "status": "success",
            "input_table": f"{dataset_id}.{table_name}",
            "output_table": f"{output_dataset}.{output_table}",
            "target_column": target_column,
            "dropped_columns": safe_columns_to_drop,
            "ignored_columns": ignored_columns
        }
    
    except Exception as e:
        logger.error(f"Error in leakage_mitigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


