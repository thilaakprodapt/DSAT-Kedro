"""Feature Engineering Pipeline Nodes.

LLM-powered feature engineering recommendations matching the original
DataScienceAssistantTool feature_engg_service.py.

Outputs 4 sections:
1. Feature Engineering (per-feature recommendations)
2. Imbalance Analysis (dataset level)
3. Bias Detection (dataset level)
4. Balancing Recommendations (dataset level)
"""

import json
import logging
from typing import Dict, List, Any, Optional

from dsat.common.llm_utils import call_gemini, parse_llm_json

logger = logging.getLogger(__name__)


def get_fe_recommendations(
    eda_summary: Dict[str, Any],
    target_column: str,
    project_id: str,
) -> Dict[str, Any]:
    """Generate feature engineering recommendations using Gemini.

    Matches the original DataScienceAssistantTool FE prompt exactly.
    Returns 4 sections: feature_engineering, imbalance_analysis,
    bias_detection, balancing_recommendations.

    Falls back to rule-based recommendations if LLM fails.

    Args:
        eda_summary: Complete EDA summary from EDA pipeline
        target_column: Fixed target column name
        project_id: GCP project ID (for Vertex AI init)

    Returns:
        Dict with feature_engineering, imbalance_analysis,
        bias_detection, and balancing_recommendations
    """
    prompt = f"""
You are a senior data scientist.

You are given an EDA report for a dataset:
{json.dumps(eda_summary)}
TARGET COLUMN (FIXED):
The target column is "{target_column}".
This column MUST be used for imbalance analysis.
Do NOT change or infer another target column.
Your task has FOUR parts:

1. FEATURE ENGINEERING (per feature)
- Handle missing values
- Scaling or transformation
- Encoding for categorical features
- Feature creation
- Feature selection or dropping
- Keep explanations short

2. IMBALANCE ANALYSIS (dataset level)
- Identify the target column
- Detect if class imbalance exists
- Calculate imbalance severity
- Explain reasoning briefly

3. BIAS DETECTION (dataset level)
- Identify sensitive features (e.g. gender, region, country, age)
- Detect representation or outcome bias
- Assign risk level with short evidence

4. BALANCING RECOMMENDATIONS (dataset level)
- If target is categorical → recommend Python Random Oversampling
- If target is continuous → recommend SMOTER
- Do NOT generate code
- Only recommend methods with pros & cons

TARGET COLUMN (IMPORTANT):
- The target column is FIXED and PROVIDED by the user.
- Target column name: "{target_column}"
- You MUST use this exact column for imbalance analysis.
- Do NOT infer or change the target column.


STRICT RULES:
- imbalance_analysis MUST always contain a valid object.
- bias_detection MUST always contain at least one feature OR a note saying "no significant bias detected".
- balancing_recommendations MUST always be present.



IMPORTANT:
- Return ONLY valid JSON
- Follow EXACTLY this output structure:
CRITICAL RULES:

1. Identify majority class = class with MAX count
2. Identify minority class = class with MIN count
3. Calculate:
   imbalance_ratio = majority_class_count / minority_class_count

4. If majority_count == minority_count:
   imbalance_ratio = 1.0
   is_imbalanced = false

5. imbalance_ratio MUST be >= 1 by definition

{{
  "feature_engineering": [
    {{
      "feature_name": "",
      "eda_insights": {{
        "distribution": [],
        "missing_values": [],
        "correlations": [],
        "outliers": []
      }},
      "selected_inputs": [],
      "recommendations": [
        {{
          "technique": "",
          "reason": ""
        }}
      ]
    }}
  ],
  "imbalance_analysis": {{
    "target_column": "",
    "target_type": "",
    "class_distribution": [],
    "imbalance_ratio": 0,
    "is_imbalanced": false,
    "severity": "",
    "reasoning": ""
  }},
  "bias_detection": [
    {{
      "sensitive_feature": "",
      "bias_type": "",
      "risk_level": "",
      "evidence": ""
    }}
  ],
  "balancing_recommendations": {{
    "recommended": true,
    "techniques": [
      {{
        "method": "",
        "applicable_when": "",
        "pros": "",
        "cons": ""
      }}
    ],
    "user_action_required": "Do you want to proceed with a balanced dataset?"
  }}
}}

Do not add explanations outside JSON.
"""

    response_text = call_gemini(prompt, project_id=project_id)
    logger.info(f"Gemini FE response: {response_text[:300]}...")

    data = parse_llm_json(response_text)

    if data is not None:
        logger.info("Successfully parsed Gemini FE recommendations")
        # Ensure all 4 sections are present
        if "feature_engineering" not in data:
            data["feature_engineering"] = []
        if "imbalance_analysis" not in data:
            data["imbalance_analysis"] = {}
        if "bias_detection" not in data:
            data["bias_detection"] = []
        if "balancing_recommendations" not in data:
            data["balancing_recommendations"] = {}
        return data
    else:
        logger.warning("Gemini parsing failed, using rule-based fallback")
        return _rule_based_recommendations(eda_summary, target_column)


def _rule_based_recommendations(
    eda_summary: Dict[str, Any],
    target_column: str,
) -> Dict[str, Any]:
    """Fallback: rule-based FE recommendations when LLM fails.

    Args:
        eda_summary: Complete EDA summary
        target_column: Fixed target column name

    Returns:
        Dict with feature_engineering, imbalance_analysis,
        bias_detection, balancing_recommendations
    """
    feature_engineering = []

    # Extract column types from EDA summary — handle both old and new format
    column_types = eda_summary.get("Column Types", {})
    if not column_types:
        data_overview = eda_summary.get("Data Overview", {})
        feature_types = data_overview.get("feature_types", {})
        column_types = {
            "numerical": feature_types.get("numerical", []),
            "categorical": feature_types.get("categorical", []),
        }

    numerical_cols = column_types.get("numerical", [])
    categorical_cols = column_types.get("categorical", [])

    # Missing value info
    data_quality = eda_summary.get("Data quality", eda_summary.get("Data Quality", {}))
    missing_values = data_quality.get("missing_values", [])
    if isinstance(missing_values, dict):
        missing_values = missing_values.get("missing_values", [])

    missing_map = {m.get("column"): m.get("missing_pct", 0) for m in missing_values if isinstance(m, dict)}

    for col in numerical_cols:
        if col == target_column:
            continue

        recs = []
        if col in missing_map and missing_map[col] > 0:
            recs.append({"technique": "median_imputation", "reason": f"{missing_map[col]}% missing"})
        recs.append({"technique": "standardization", "reason": "Numerical column - standard scaling for ML"})

        feature_engineering.append({
            "feature_name": col,
            "eda_insights": {
                "distribution": [],
                "missing_values": [f"{missing_map.get(col, 0)}% missing"],
                "correlations": [],
                "outliers": [],
            },
            "selected_inputs": [],
            "recommendations": recs,
        })

    for col in categorical_cols:
        if col == target_column:
            continue

        recs = []
        if col in missing_map and missing_map[col] > 0:
            recs.append({"technique": "mode_imputation", "reason": f"{missing_map[col]}% missing"})
        recs.append({"technique": "label_encoding", "reason": "Categorical column needs encoding for ML"})

        feature_engineering.append({
            "feature_name": col,
            "eda_insights": {
                "distribution": [],
                "missing_values": [f"{missing_map.get(col, 0)}% missing"],
                "correlations": [],
                "outliers": [],
            },
            "selected_inputs": [],
            "recommendations": recs,
        })

    return {
        "feature_engineering": feature_engineering,
        "imbalance_analysis": {
            "target_column": target_column,
            "target_type": "unknown",
            "class_distribution": [],
            "imbalance_ratio": 1.0,
            "is_imbalanced": False,
            "severity": "unknown",
            "reasoning": "LLM unavailable — run with LLM for accurate analysis",
        },
        "bias_detection": [
            {
                "sensitive_feature": "unknown",
                "bias_type": "unknown",
                "risk_level": "unknown",
                "evidence": "LLM unavailable — run with LLM for accurate analysis",
            }
        ],
        "balancing_recommendations": {
            "recommended": False,
            "techniques": [],
            "user_action_required": "Re-run with LLM enabled for recommendations",
        },
    }


def generate_dag_code(
    recommendations: Dict[str, Any],
    project_id: str,
    dataset_id: str,
    table_name: str,
    target_dataset: str,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """Generate Airflow DAG code from recommendations.

    Args:
        recommendations: FE recommendations dict (with feature_engineering key)
        project_id: GCP project ID
        dataset_id: Source dataset ID
        table_name: Source table name
        target_dataset: Target dataset for transformed data
        target_column: Optional target column

    Returns:
        Dict with DAG code and metadata
    """
    from dsat.common import DAGGenerator

    generator = DAGGenerator(
        project_id=project_id,
        dataset_id=dataset_id,
        source_table=table_name,
        target_dataset=target_dataset
    )

    # Extract transformations from the feature_engineering section
    fe_list = recommendations.get("feature_engineering", [])
    transformations = []
    for feature in fe_list:
        feature_name = feature.get("feature_name", "")
        for rec in feature.get("recommendations", []):
            technique = rec.get("technique", "")
            if technique and feature_name:
                transformations.append({
                    "column_name": feature_name,
                    "fe_method": technique
                })

    if not transformations:
        logger.warning("No transformations extracted from recommendations")
        return {
            "status": "no_transformations",
            "dag_code": "",
            "message": "No valid transformations found in recommendations"
        }

    result = generator.generate(transformations, target_column=target_column)

    logger.info(f"Generated DAG: {result.get('dag_id', 'unknown')}")
    return result


def preview_sql(dag_result: Dict[str, Any]) -> str:
    """Extract and return the SQL from the DAG.

    Args:
        dag_result: Result from generate_dag_code

    Returns:
        SQL query string
    """
    import re

    dag_code = dag_result.get("dag_code", "")
    match = re.search(r'"""(CREATE OR REPLACE.*?)"""', dag_code, re.DOTALL)

    return match.group(1).strip() if match else "SQL extraction failed"
