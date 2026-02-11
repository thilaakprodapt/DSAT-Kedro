"""ML model helper utilities."""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# Initialize BigQuery client with service account
def _get_credentials():
    """Get credentials from service account file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        return service_account.Credentials.from_service_account_file(credentials_path)
    return None

credentials = _get_credentials()
bq_client = bigquery.Client(credentials=credentials) if credentials else bigquery.Client()


def build_basic_metadata(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Build basic metadata about the dataset for ML context generation.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of the target column
        
    Returns:
        dict: Metadata including row count, column count, problem type, etc.
    """
    feature_types = df.dtypes.astype(str).to_dict()

    class_distribution = {}
    problem_type = "regression"

    if df[target_column].nunique() < 20:
        problem_type = "classification"
        class_distribution = (
            df[target_column].value_counts(normalize=True).to_dict()
        )

    return {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "target_column": target_column,
        "problem_type": problem_type,
        "class_distribution": class_distribution,
        "feature_types": feature_types
    }


def convert_numpy(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def classification_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics including confusion matrix breakdown.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class
        
    Returns:
        dict: Classification metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):   # Binary classification
        tn, fp, fn, tp = cm.ravel()
    else:  # Multi-class
        tn = fp = fn = tp = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),

        # 🔥 Confusion matrix breakdown
        "true_negative": int(tn) if tn is not None else None,
        "false_positive": int(fp) if fp is not None else None,
        "false_negative": int(fn) if fn is not None else None,
        "true_positive": int(tp) if tp is not None else None
    }


def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Regression metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred))
    }


def detect_problem_type(y):
    """
    Detect whether the problem is classification or regression.
    
    Args:
        y: Target variable series
        
    Returns:
        str: "classification" or "regression"
    """
    if str(y.dtype) == "object":
        return "classification"
    if y.nunique() <= 20 and sorted(y.unique()) == list(range(len(y.unique()))):
        return "classification"
    return "regression"


def get_classification_model(model_name):
    """
    Get classification model instance by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Sklearn-compatible classifier or None
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(eval_metric="logloss"),
        "lightgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0),
        "svm": SVC(probability=True),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "deep_neural_network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    }
    return models.get(model_name)


def get_regression_model(model_name):
    """
    Get regression model instance by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Sklearn-compatible regressor or None
    """
    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(),
        "lasso_regression": Lasso(),
        "decision_tree_regressor": DecisionTreeRegressor(),
        "random_forest_regressor": RandomForestRegressor(),
        "xgboost_regressor": XGBRegressor(),
        "lightgbm_regressor": LGBMRegressor(),
        "catboost_regressor": CatBoostRegressor(verbose=0),
        "svr": SVR(),
        "deep_neural_network_regressor": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    }
    return models.get(model_name)


def get_model(model_name, problem_type):
    """
    Get model instance based on name and problem type.
    
    Args:
        model_name: Name of the model
        problem_type: "classification" or "regression"
        
    Returns:
        Sklearn-compatible model or None
    """
    if problem_type == "classification":
        return get_classification_model(model_name)
    else:
        return get_regression_model(model_name)


def load_table_from_bigquery(dataset_id, table_name):
    """
    Load table from BigQuery as DataFrame.
    
    Args:
        dataset_id: BigQuery dataset ID (format: project.dataset)
        table_name: Name of the table
        
    Returns:
        pd.DataFrame: Loaded data
    """
    query = f"SELECT * FROM `{dataset_id}.{table_name}`"
    return bq_client.query(query).to_dataframe()

