"""Data Balancing API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import pandas as pd
import numpy as np
import json
import re
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel

from dsat.common.imbalance_utils import (
    check_imbalance,
    check_continuous_imbalance,
    deep_flatten_and_convert,
    clean_target_column,
    find_target_column,
    bq_client,
)

from dsat.common.balancing_techniques import (
    run_imbalance_analysis,
    pandas_random_oversample,
    pandas_random_undersample,
    smote_oversample,
    smotenc_resample,
    adasyn_oversample,
    cluster_based_oversample,
    smoter_smogn,
    gaussian_noise_injection,
    kde_resample,
    quantile_binning_oversample,
    tail_focused_resampling,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class BalanceRequest(BaseModel):
    """Request model for data balancing."""
    dataset_id: str
    table_name: str
    technique: str


class ImbalanceRequest(BaseModel):
    """Request model for imbalance analysis."""
    dataset_id: str
    table_name: str

@router.post("/imbalance_analysis")
async def imbalance_analysis(request: ImbalanceRequest):
    """
    Analyze imbalance on transformed data.
    Imbalance detection -> Python
    Recommendations -> LLM
    """
    print("🔥🔥🔥 IMBALANCE ENDPOINT HIT 🔥🔥🔥")
    
    try:
        # -------------------------------------------------
        # 1. Load transformed data
        # -------------------------------------------------
        source_table = request.table_name

        query = f"""
        SELECT *
        FROM `{request.dataset_id}.{source_table}`
        """
        df = bq_client.query(query).to_dataframe()
        print(f"Loaded transformed data: {df.shape}")

        # -------------------------------------------------
        # 2. Find target column
        # -------------------------------------------------
        target_column = find_target_column(request.dataset_id, source_table)

        if target_column not in df.columns:
            raise HTTPException(400, "Target column not found in source table")
        
        # -------------------------------------------------
        # Deep flatten target column and convert to native Python types
        # -------------------------------------------------
        df[target_column] = df[target_column].apply(deep_flatten_and_convert)
        df, target_cleaning_metadata = clean_target_column(df, target_column)

        print("Target cleaning:", target_cleaning_metadata)

        # -------------------------------------------------
        # 3. Detect target type (Python)
        # -------------------------------------------------
        unique_vals = df[target_column].dropna().unique()
        num_unique = len(unique_vals)

        if pd.api.types.is_numeric_dtype(df[target_column]):
            if num_unique <= 10 or set(unique_vals).issubset({0, 1}):
                target_type = "categorical"
            else:
                target_type = "continuous"
        else:
            target_type = "categorical"

        print("Target type:", target_type)
        
        # -------------------------------------------------
        # 4. Check imbalance (Python ONLY)
        # -------------------------------------------------
        is_imbalanced = False
        class_distribution = {}

        if target_type == "categorical":
            imbalance_result = check_imbalance(df[target_column])
            status_label = imbalance_result["status"]
            is_imbalanced = status_label != "balanced"
            class_distribution = imbalance_result["class_distribution"]
            imbalance_metadata = {
                "minority_ratio": imbalance_result["minority_ratio"],
                "severity": status_label
            }
        else:  # continuous
            imbalance_result = check_continuous_imbalance(df[target_column])
            is_imbalanced = imbalance_result["is_imbalanced"]
            imbalance_metadata = {
                "skewness": imbalance_result.get("skewness"),
                "rare_ratio": imbalance_result.get("rare_ratio")
            }

        status = "Imbalance" if is_imbalanced else "Balance"

        # -------------------------------------------------
        # 5. If balanced → return immediately
        # -------------------------------------------------
        if not is_imbalanced:
            return {
                "target_type": target_type,
                "status": status,
                "target_cleaning": target_cleaning_metadata,
                "imbalance_details": (
                    {
                        "class_distribution": class_distribution,
                        **imbalance_metadata
                    }
                    if target_type == "categorical"
                    else imbalance_metadata
                ),
                "techniques": []
            }
        
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        feature_metadata = {
            "numerical_feature_count": len(numerical_cols),
            "categorical_feature_count": len(categorical_cols),
            "has_mixed_features": len(numerical_cols) > 0 and len(categorical_cols) > 0
        }
       
        # -------------------------------------------------
        # 6. LLM for recommendations ONLY
        # -------------------------------------------------
        model = GenerativeModel("gemini-2.5-pro")

        prompt = f"""
You are an expert machine learning engineer.

You MUST strictly follow the rules below.
If any rule is violated, the response is INVALID.

--------------------------------------------------
AUTHORITATIVE METADATA (Computed in Python)
--------------------------------------------------
Target column: {target_column}
Target type: {target_type}

Feature metadata:
{json.dumps(feature_metadata)}

Imbalance diagnostics:
{json.dumps(
    class_distribution if target_type == "categorical"
    else imbalance_metadata
)}

--------------------------------------------------
DECISION RULES (MANDATORY)
--------------------------------------------------

IF target_type == "continuous":
- Recommend ONLY techniques suitable for regression imbalance
Allowed techniques:
- SMOTER
- GAUSSIAN NOISE INJECTION
- KDE-BASED RESAMPLING
- QUANTILE BASED BINNING+ OVERSAMPLING
- TAIL-FOCUSED RESAMPLING

IF target_type == "categorical":
- Read feature_metadata.has_mixed_features
- Do NOT infer feature types yourself

--------------------------------------------------
TECHNIQUE RECOMMENDATION RULES
--------------------------------------------------

IF target_type == "categorical" AND has_mixed_features == true:
- Recommend ONLY:
  - SMOTENC
  - Random Oversampling
  - Random Undersampling
  - Cluster-Based Oversampling
- Do NOT recommend:
  - SMOTE
  - ADASYN

IF target_type == "categorical" AND has_mixed_features == false:
- Recommend ONLY:
  - SMOTE
  - ADASYN
  - Random Oversampling
  - Random Undersampling
  - Cluster-Based Oversampling

--------------------------------------------------
STRICT OUTPUT FORMAT (NO EXTRA TEXT)
--------------------------------------------------
{{
  "target_type": "",
  "has_mixed_features": null,
  "techniques": []
}}
"""       
        response = model.generate_content(prompt)
        response_text = response.text
        
        # -------------------------------------------------
        # 7. Parse LLM response
        # -------------------------------------------------
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        techniques = []

        if json_match:
            parsed = json.loads(json_match.group(0))
            techniques = parsed.get("techniques", [])

        # -------------------------------------------------
        # 8. Final response
        # -------------------------------------------------
        return {
            "target_type": target_type,
            "status": status,
            "imbalance_details": (
                {
                    "class_distribution": class_distribution,
                    **imbalance_metadata
                }
                if target_type == "categorical"
                else imbalance_metadata
            ),
            "techniques": techniques
        }

    except Exception as e:
        print(f"Error in imbalance_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/balance_data")
async def balance_data(request: BalanceRequest) -> Dict[str, Any]:
    """Balance imbalanced dataset using various techniques.
    
    Supports both categorical and continuous target balancing.
    Matches original endpoint: POST /DataBalancing/balance_data
    """
    try:
        print("STEP 0: balance_data started", flush=True)

        # Extract project_id from dataset_id (format: project.dataset)
        if '.' in request.dataset_id:
            project_id = request.dataset_id.split('.')[0]
        else:
            raise HTTPException(400, "dataset_id must be in format 'project.dataset'")

        # 1️⃣ Load source table
        source_table = request.table_name
        query = f"SELECT * FROM `{request.dataset_id}.{source_table}`"
        df = bq_client.query(query).to_dataframe()

        if df.empty:
            raise HTTPException(400, f"Table {source_table} is empty")

        print("STEP 1: data loaded", flush=True)
        print(df.dtypes, flush=True)

        # 2️⃣ Detect target column
        target_column = find_target_column(request.dataset_id, source_table)
        
        if target_column is None:
            raise HTTPException(400, f"Target column not found in {request.dataset_id}.{source_table}")
            
        if target_column not in df.columns:
             raise HTTPException(400, f"Identified target '{target_column}' not found in dataframe columns: {df.columns.tolist()}")

        df[target_column] = df[target_column].apply(deep_flatten_and_convert)
        df, target_cleaning_metadata = clean_target_column(df, target_column)

        unique_vals = df[target_column].dropna().unique()
        if pd.api.types.is_numeric_dtype(df[target_column]) and set(unique_vals).issubset({0, 1}):
            target_type = "categorical"
        elif pd.api.types.is_numeric_dtype(df[target_column]):
            target_type = "continuous"
        else:
            target_type = "categorical"

        print(f"STEP 2: target={target_column}, type={target_type}", flush=True)
        
        categorical_cols = [
            col for col in df.columns
            if col != target_column and df[col].dtype == "object"
        ]
        
        # 3️⃣ Apply balancing
        technique = request.technique.lower()

        if target_type == "categorical":
            if technique == "random oversampling":
                balanced_df = pandas_random_oversample(df, target_column)
            elif technique == "random undersampling":
                balanced_df = pandas_random_undersample(df, target_column)
            elif technique == "smote":
                balanced_df = smote_oversample(df, target_column)
            elif technique == "smotenc":
                balanced_df = smotenc_resample(df, target_column, categorical_cols=categorical_cols)
            elif technique == "adasyn":
                balanced_df = adasyn_oversample(df, target_column)
            elif technique == "cluster-based oversampling":
                balanced_df = cluster_based_oversample(df, target_column)
            else:
                raise HTTPException(400, f"Unsupported categorical technique: {technique}")
        else:
            if technique == "smoter":
                balanced_df = smoter_smogn(df, target_column)
            elif technique == "gaussian noise injection":
                balanced_df = gaussian_noise_injection(df, target_column)
            elif technique == "kde-based resampling":
                balanced_df = kde_resample(df, target_column)
            elif technique == "quantile-based binning + oversampling":
                balanced_df = quantile_binning_oversample(df, target_column)
            elif technique == "tail-focused resampling":
                balanced_df = tail_focused_resampling(df, target_column)
            else:
                raise HTTPException(400, f"Unsupported continuous technique: {technique}")

        print("STEP 3: balancing completed", flush=True)
        print(balanced_df.head(3), flush=True)

        # ✅ STEP 3.5: FIX TARGET COLUMN (ONLY TARGET)
        if target_type == "categorical":
            balanced_df[target_column] = (
                balanced_df[target_column]
                .round()
                .astype("int64")
            )

        print("STEP 3.5: target column fixed", flush=True)
        print(balanced_df[target_column].value_counts(dropna=False), flush=True)
        
        post_balance_imbalance = run_imbalance_analysis(
            balanced_df,
            target_column
        )

        print("STEP 3.6: post-balance imbalance check", flush=True)
        print(post_balance_imbalance, flush=True)
        
        # 4️⃣ 🔥 CRITICAL FIX: force safe dtypes (EXCEPT TARGET)
        for col in balanced_df.columns:
            if col == target_column:
                continue  # 👈 VERY IMPORTANT
            if pd.api.types.is_numeric_dtype(balanced_df[col]):
                balanced_df[col] = balanced_df[col].astype("float64")
            else:
                balanced_df[col] = balanced_df[col].astype(str)
        
        print("STEP 4: dtypes normalized", flush=True)
        print(balanced_df.dtypes, flush=True)

        # 5️⃣ Build BigQuery schema (FLOAT64 only for numeric)
        schema = []
        for col in balanced_df.columns:
            if col == target_column and target_type == "categorical":
                schema.append(bigquery.SchemaField(col, "INT64"))   # 👈 FIX
            elif pd.api.types.is_numeric_dtype(balanced_df[col]):
                schema.append(bigquery.SchemaField(col, "FLOAT64"))
            else:
                schema.append(bigquery.SchemaField(col, "STRING"))

        balanced_dataset = "DS_Balance_Dataset"
        balanced_table = f"{source_table}_balanced"
        table_id = f"{project_id}.{balanced_dataset}.{balanced_table}"
        
        # Check and create dataset if needed
        dataset_ref = bq_client.dataset(balanced_dataset, project=project_id)
        try:
            bq_client.get_dataset(dataset_ref)
        except Exception:
            print(f"Dataset {balanced_dataset} not found. Creating it...", flush=True)
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # Default location, adjust if needed
            bq_client.create_dataset(dataset, exists_ok=True)

        # 6️⃣ Load to BigQuery (NO TEMP TABLE NEEDED)
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
            autodetect=False
        )

        print("STEP 5: loading to BigQuery", flush=True)

        load_job = bq_client.load_table_from_dataframe(
            balanced_df,
            table_id,
            job_config=job_config
        )
        load_job.result()

        print("STEP 6: load successful", flush=True)

        # 7️⃣ Response
        return {
            "source_table": source_table,
            "balanced_table": balanced_table,
            "target_column": target_column,
            "target_type": target_type,
            "technique_applied": technique,
            "original_shape": df.shape,
            "balanced_shape": balanced_df.shape,
            "target_cleaning": target_cleaning_metadata,
            "post_balance_imbalance_analysis": post_balance_imbalance 
        }

    except Exception as e:
        import traceback
        print("❌ ERROR in balance_data", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
