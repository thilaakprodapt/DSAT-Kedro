"""ML Model recommendation API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import pandas as pd
import json
import re
import os
import joblib
from vertexai.generative_models import GenerativeModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dsat.common.imbalance_utils import (
    find_target_column,
    bq_client,
)

from dsat.common.ml_utils import (
    build_basic_metadata,
    convert_numpy,
    classification_metrics,
    regression_metrics,
    detect_problem_type,
    get_model,
    load_table_from_bigquery,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ContextGenerationRequest(BaseModel):
    """Request model for dataset context generation."""
    dataset_id: str
    table_name: str
    feature_engineering_steps: Optional[List[Dict[str, Any]]] = []
    balancing_applied: Optional[bool] = False
    balancing_method: Optional[str] = None
    leakage_columns_removed: Optional[List[str]] = []


class ModelRecommendationRequest(BaseModel):
    """Request model for model recommendation."""
    dataset_id: str
    table_name: str
    context_file: Dict[str, Any]


class TrainModelRequest(BaseModel):
    """Request model for model training."""
    dataset_id: str
    table_name: str
    model_name: str


@router.post("/generate-dataset-context")
def generate_dataset_context(request: ContextGenerationRequest):
    """
    Generate ML dataset context using LLM analysis.
    
    Analyzes dataset metadata and generates structured context file
    including dataset profile, target analysis, feature analysis, and risk flags.
    """
    try:
        print(f"STEP 0: Context verification started for table: {request.table_name}", flush=True)
        print(f"Request data: {request.dict()}", flush=True)

        # Handle dataset_id format (project.dataset vs dataset only)
        if '.' in request.dataset_id:
            project_id = request.dataset_id.split('.')[0]
            dataset_only = request.dataset_id.split('.')[1]
        else:
            # Fallback to default project_id if only dataset name provided
            project_id = bq_client.project
            dataset_only = request.dataset_id

        full_dataset_id = f"{project_id}.{dataset_only}"
        
        # 🔹 Step 1: Load processed table
        table_name = f"{request.table_name}"
        query = f"SELECT * FROM `{full_dataset_id}.{table_name}`"
        df = bq_client.query(query).to_dataframe()

        target_column = find_target_column(full_dataset_id, table_name)
        
        if target_column is None:
             raise HTTPException(400, f"Target column not found in {full_dataset_id}.{table_name}")

        # 🔹 Step 2: Build safe metadata (NO raw data sent to LLM)
        basic_metadata = build_basic_metadata(df, target_column)

        metadata_payload = {
            "dataset_id": full_dataset_id,
            "table_name": request.table_name,
            "basic_metadata": basic_metadata,
            "feature_engineering_steps": request.feature_engineering_steps,
            "balancing_applied": request.balancing_applied,
            "balancing_method": request.balancing_method,
            "leakage_columns_removed": request.leakage_columns_removed
        }
        
        model = GenerativeModel("gemini-2.0-flash") # Using 1.0-pro for stability
        safe_metadata = convert_numpy(metadata_payload)
        
        # 🧠 Step 3: LLM reasoning via Vertex AI Gemini
        prompt = f"""
        You are an expert ML system assistant.
        
        Using the dataset metadata below, generate a structured ML context file.
        Describe dataset characteristics, target analysis, feature analysis, and data risk flags.
        
        Do NOT include raw data. Only reason using metadata.
        
        Metadata:
        {json.dumps(safe_metadata, indent=2)}
        
        Output JSON format (STRICT):
        {{
          "dataset_profile": {{
            "num_rows": ,
            "num_columns": ,
            "feature_types": {{}},
            "missing_values": ,
            "high_cardinality_categoricals": []
          }},
          "target_analysis": {{
            "target_column": "",
            "problem_type": "",
            "num_classes": ,
            "class_distribution": {{}},
            "imbalanced": ,
            "imbalance_ratio": 
          }},
          "feature_analysis": {{
            "numerical_features": [],
            "categorical_features": [],
            "binary_features": [],
            "text_features": [],
            "datetime_features": [],
            "skewed_features": [],
            "outliers_detected": []
          }},
          "data_risk_flags": {{
            "missing_values_present": ,
            "outliers_present": ,
            "leakage_columns_removed": []
          }}
        }}
        """
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 8192
            }
        )
        llm_response_text = response.text.strip()

        # ------------------------------
        # Inline regex JSON extraction
        # ------------------------------
        # Step 1: Remove ```json and ``` if present
        cleaned_text = llm_response_text
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)

        # Step 2: Extract JSON between outermost braces
        match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        if match:
            try:
                context_file = json.loads(match.group(1))
            except json.JSONDecodeError:
                context_file = None
        else:
            context_file = None

        if context_file is None:
            return {
                "status": "error",
                "message": "LLM did not return valid JSON",
                "raw_response": llm_response_text
            }
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "table_name": table_name,
            "target_column": target_column,
            **context_file
        }
    
    except Exception as e:
        logger.error(f"Error in generate_dataset_context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-recommendation")
def model_recommendation(request: ModelRecommendationRequest):
    """
    Recommend ML models based on dataset context.
    
    Uses LLM to analyze dataset characteristics and recommend
    4-8 appropriate ML models with reasoning.
    """
    try:
        context = request.context_file or {}

        dataset_profile = context.get("dataset_profile") or {}
        target_analysis = context.get("target_analysis") or {}
        feature_analysis = context.get("feature_analysis") or {}
        data_risk_flags = context.get("data_risk_flags") or {}

        problem_type = str(target_analysis.get("problem_type", "")).lower().strip()
        target_column = target_analysis.get("target_column")

        # 🔒 STRICT VALIDATION
        if problem_type not in ["classification", "regression"]:
            raise ValueError("Invalid or missing problem_type in context file")

        if not target_column:
            raise ValueError("Target column missing in context file")
        
        clean_context = {
            "dataset_size": dataset_profile.get("num_rows"),
            "num_features": dataset_profile.get("num_columns"),
            "high_cardinality_categoricals": dataset_profile.get("high_cardinality_categoricals", []),

            "problem_type": problem_type,
            "num_classes": target_analysis.get("num_classes"),
            "imbalanced": target_analysis.get("imbalanced"),
            "imbalance_ratio": target_analysis.get("imbalance_ratio"),

            "num_numerical_features": len(feature_analysis.get("numerical_features", [])),
            "num_categorical_features": len(feature_analysis.get("categorical_features", [])),
            "num_text_features": len(feature_analysis.get("text_features", [])),
            "outliers_detected": feature_analysis.get("outliers_detected", []),

            "risk_flags": data_risk_flags
        }

        # ---------------------------------------------------------
        # 🧠 BUILD INTELLIGENT PROMPT FROM CONTEXT FILE
        # ---------------------------------------------------------
        prompt = f"""
You are an expert ML architect designing a training strategy.

Use ONLY the dataset context below. Do NOT assume anything not mentioned.

DATASET CONTEXT:
{json.dumps(clean_context, indent=2)}

Your job:
Recommend the BEST ML models for training considering:

• Dataset size  
• Feature types  
• Class imbalance  
• High cardinality features  
• Outliers  
• Risk flags  

You MUST:
1. Recommend between 4 and 8 models
2. Include BOTH baseline and advanced models
3. Choose models that fit THIS dataset (not generic)
4. Do NOT invent model names
5. Output STRICT JSON ONLY

Allowed Classification Models:
logistic_regression  
decision_tree  
random_forest  
xgboost  
lightgbm  
catboost  
svm  
knn  
naive_bayes  
deep_neural_network  

Allowed Regression Models:
linear_regression  
ridge_regression  
lasso_regression  
decision_tree_regressor  
random_forest_regressor  
xgboost_regressor  
lightgbm_regressor  
catboost_regressor  
svr  
deep_neural_network_regressor  

Output Format:
{{
  "recommended_models": [
    {{
      "model_name": "",
      "reason": ""
    }}
  ]
}}
"""

        # ---------------------------------------------------------
        # 🤖 LLM CALL
        # ---------------------------------------------------------
        model = GenerativeModel("gemini-2.5-pro")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 4096
            }
        )
        print("CONTEXT RECEIVED:", request.context_file)
        llm_text = response.text.strip()

        # ---------------------------------------------------------
        # 🧹 CLEAN LLM RESPONSE
        # ---------------------------------------------------------
        llm_text = re.sub(r'^```json\s*', '', llm_text)
        llm_text = re.sub(r'\s*```$', '', llm_text)

        match = re.search(r'(\{.*\})', llm_text, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return valid JSON")

        result = json.loads(match.group(1))
        models = result.get("recommended_models", [])

        # ---------------------------------------------------------
        # 🛡 VALIDATION
        # ---------------------------------------------------------
        if not (4 <= len(models) <= 8):
            raise ValueError("LLM must recommend between 4 and 8 models")

        allowed_classification = {
            "logistic_regression", "decision_tree", "random_forest", "xgboost",
            "lightgbm", "catboost", "svm", "knn", "naive_bayes", "deep_neural_network"
        }

        allowed_regression = {
            "linear_regression", "ridge_regression", "lasso_regression",
            "decision_tree_regressor", "random_forest_regressor", "xgboost_regressor",
            "lightgbm_regressor", "catboost_regressor", "svr", "deep_neural_network_regressor"
        }

        for m in models:
            if "model_name" not in m or "reason" not in m:
                raise ValueError("Invalid model format from LLM")

            name = m["model_name"]

            if problem_type == "classification" and name not in allowed_classification:
                raise ValueError(f"Invalid classification model: {name}")

            if problem_type == "regression" and name not in allowed_regression:
                raise ValueError(f"Invalid regression model: {name}")

        # ---------------------------------------------------------
        # 🎯 FINAL RESPONSE
        # ---------------------------------------------------------
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "table_name": request.table_name,
            "target_column": target_column,
            "problem_type": problem_type,
            "recommended_models": models
        }

    except Exception as e:
        logger.error(f"Error in model_recommendation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train-model")
def train_model(request: TrainModelRequest):
    """
    Train ML model on dataset with automatic problem type detection.
    
    Performs train-test split, optional scaling, model training,
    evaluation, and saves the trained model to disk.
    """
    try:
        # Handle dataset_id format (project.dataset vs dataset only)
        if '.' in request.dataset_id:
            project_id = request.dataset_id.split('.')[0]
            dataset_only = request.dataset_id.split('.')[1]
        else:
            # Fallback to default project_id if only dataset name provided
            project_id = bq_client.project
            dataset_only = request.dataset_id

        full_dataset_id = f"{project_id}.{dataset_only}"

        # 1️⃣ Load dataset
        df = load_table_from_bigquery(full_dataset_id, request.table_name)

        # 2️⃣ Auto-detect target column
        target_column = find_target_column(full_dataset_id, request.table_name)

        if target_column is None:
             raise HTTPException(400, f"Target column not found in {full_dataset_id}.{request.table_name}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 3️⃣ Detect problem type
        problem_type = detect_problem_type(y)

        # 4️⃣ Train-test split
        if problem_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # 5️⃣ Load model
        model = get_model(request.model_name, problem_type)
        if model is None:
            raise HTTPException(status_code=400, detail="Invalid model selection")

        # 6️⃣ Scaling if required
        SCALING_MODELS = [
            "logistic_regression", "svm", "knn", "svr",
            "linear_regression", "ridge_regression", "lasso_regression",
            "deep_neural_network", "deep_neural_network_regressor"
        ]

        scaler_used = False
        scaler = None
        if request.model_name in SCALING_MODELS:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            scaler_used = True

        # 7️⃣ Train model
        model.fit(X_train, y_train)

        # 8️⃣ Predictions
        y_pred = model.predict(X_test)

        # 9️⃣ Evaluation
        if problem_type == "classification":
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = regression_metrics(y_test, y_pred)

        # 🔟 Bias/Variance check
        train_score = float(model.score(X_train, y_train))
        test_score = float(model.score(X_test, y_test))

        if train_score < 0.7 and test_score < 0.7:
            model_behavior = "underfitting"
        elif train_score > 0.9 and test_score < 0.75:
            model_behavior = "overfitting"
        else:
            model_behavior = "good_fit"

        # 1️⃣1️⃣ Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{request.model_name}_{target_column}.pkl"
        joblib.dump(model, model_path)
        
        if scaler_used:
            scaler_path = f"models/{request.model_name}_{target_column}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
        else:
            scaler_path = None
            
        # 1️⃣2️⃣ Return raw structured output (LLM can format later)
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "table_name": request.table_name,
            "target_column": target_column,
            "problem_type": problem_type,
            "model_name": request.model_name,
            "metrics": metrics,
            "train_score": train_score,
            "test_score": test_score,
            "model_behavior": model_behavior,
            "scaler_used": scaler_used,
            "model_saved_path": model_path,
            "scaler_saved_path": scaler_path
        }

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
