"""
DAG Generator using SQL Templates - ML-Ready Output (CTE-based)

This module generates Airflow DAG code that produces ML-ready transformed data:
- Uses CTEs to avoid nested analytic functions (BigQuery limitation)
- Chains transformations properly (impute → encode)
- Outputs only final columns (no originals, no intermediate)
- Ensures no NULLs in output
- All features are numeric
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dsat.common.sql_templates import SQLTemplateEngine, validate_column_name, sanitize_column_name


class DAGGenerator:
    """
    Generates Airflow DAG Python code with ML-ready transformations.
    Uses CTE-based SQL to avoid nested analytic functions.
    """
    
    DAG_TEMPLATE = '''"""
Auto-generated Airflow DAG for Feature Engineering
Generated at: {generated_at}
Source: {source_table}
Target: {target_table}
"""

from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# DAG CONFIGURATION
# -----------------------------------------------------------------------------

PROJECT_ID = "{project_id}"
SOURCE_DATASET = "{source_dataset}"
SOURCE_TABLE = "{source_table}"
TARGET_DATASET = "{target_dataset}"
TARGET_TABLE = "{target_table}"

default_args = {{
    'owner': 'data-science-assistant',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}}

# -----------------------------------------------------------------------------
# DAG DEFINITION
# -----------------------------------------------------------------------------

with DAG(
    dag_id='{dag_id}',
    default_args=default_args,
    description='Feature Engineering DAG for {source_table}',
    schedule_interval=None,  # Triggered manually
    catchup=False,
    max_active_runs=1,
    tags=['feature-engineering', 'auto-generated', 'ml-ready'],
) as dag:

{task_definitions}

{task_dependencies}
'''

    TASK_TEMPLATE = '''
    # -------------------------------------------------------------------------
    # Task: {task_id}
    # {description}
    # -------------------------------------------------------------------------
    {task_id} = BigQueryInsertJobOperator(
        task_id='{task_id}',
        configuration={{
            "query": {{
                "query": """{sql_query}""",
                "useLegacySql": False,
                "priority": "BATCH",
            }}
        }},
        location='US',
    )
'''

    # Transformation priority for chaining order
    TRANSFORM_PRIORITY = {
        # Imputation comes first (priority 1)
        "impute_mean": 1, "impute_median": 1, "impute_mode": 1, "impute_constant": 1,
        "mean_imputation": 1, "median_imputation": 1, "mode_imputation": 1,
        
        # Scaling/transformation comes second (priority 2)
        "standardization": 2, "normalization": 2, "min_max_normalization": 2,
        "log_transformation": 2, "sqrt_transformation": 2,
        "winsorize": 2, "clip_outliers": 2, "clip_iqr": 2, "robust_scaling": 2,
        
        # Encoding comes last (priority 3)
        "label_encoding": 3, "frequency_encoding": 3, "hash_encoding": 3,
        "binary_encoding": 3, "one_hot_encoding": 3, "target_encoding": 3,
    }

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        source_table: str,
        target_dataset: str,
        target_table: Optional[str] = None
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.source_table = source_table
        self.target_dataset = target_dataset
        self.target_table = target_table or f"{source_table}_transformed"
        
        # Generate unique DAG ID
        self.dag_id = f"fe_dag_{source_table}_{uuid.uuid4().hex[:8]}"
        
        # Initialize SQL template engine
        self.sql_engine = SQLTemplateEngine(project_id, dataset_id, source_table)
    
    def _normalize_method(self, method: str) -> str:
        """Normalize method name for consistency."""
        return method.lower().replace(" ", "_").replace("-", "_")
    
    def _get_transform_priority(self, method: str) -> int:
        """Get priority for a transformation method (lower = earlier)."""
        normalized = self._normalize_method(method)
        return self.TRANSFORM_PRIORITY.get(normalized, 2)
    
    def _is_imputation(self, method: str) -> bool:
        """Check if method is an imputation type."""
        normalized = self._normalize_method(method)
        return "impute" in normalized or "imputation" in normalized
    
    def _is_encoding(self, method: str) -> bool:
        """Check if method is an encoding type."""
        normalized = self._normalize_method(method)
        return "encoding" in normalized or "encode" in normalized
    
    def _is_scaling(self, method: str) -> bool:
        """Check if method is a scaling/normalization type."""
        normalized = self._normalize_method(method)
        return any(x in normalized for x in ["standard", "normal", "min_max", "scale", "robust"])
    
    def _group_transformations_by_column(
        self, 
        transformations: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group transformations by column and sort by priority."""
        grouped = defaultdict(list)
        
        for t in transformations:
            column = t.get("column_name", "")
            if column:
                grouped[column].append(t)
        
        for column in grouped:
            grouped[column].sort(key=lambda x: self._get_transform_priority(x.get("fe_method", "")))
        
        return dict(grouped)
    
    def _render_imputation_expr(self, column: str, method: str, params: dict) -> str:
        """Render SQL for imputation (Stage 1)."""
        normalized = self._normalize_method(method)
        
        if "median" in normalized:
            # Use subquery for median since PERCENTILE_CONT needs aggregation
            return f"COALESCE({column}, (SELECT APPROX_QUANTILES({column}, 2)[OFFSET(1)] FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`))"
        elif "mean" in normalized:
            return f"COALESCE({column}, (SELECT AVG({column}) FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`))"
        elif "mode" in normalized:
            # Use subquery to find mode
            return f"COALESCE({column}, (SELECT {column} FROM `{self.project_id}.{self.dataset_id}.{self.source_table}` WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT 1))"
        elif "constant" in normalized:
            fill_value = params.get("fill_value", 0)
            if isinstance(fill_value, str):
                return f"COALESCE({column}, '{fill_value}')"
            return f"COALESCE({column}, {fill_value})"
        
        return column
    
    def _render_encoding_expr(self, column: str, method: str, params: dict) -> str:
        """Render SQL for encoding (Stage 2) - operates on imputed column."""
        normalized = self._normalize_method(method)
        
        if "hash" in normalized or "one_hot" in normalized:
            num_buckets = params.get("num_buckets", 50)
            return f"MOD(ABS(FARM_FINGERPRINT(CAST({column} AS STRING))), {num_buckets})"
        elif "binary" in normalized:
            return f"CASE WHEN CAST({column} AS STRING) IN ('Yes', 'yes', 'YES', 'true', 'True', 'TRUE', '1') THEN 1 ELSE 0 END"
        elif "label" in normalized:
            # Use DENSE_RANK for label encoding
            return f"DENSE_RANK() OVER (ORDER BY CAST({column} AS STRING)) - 1"
        elif "frequency" in normalized:
            return f"COUNT(*) OVER (PARTITION BY CAST({column} AS STRING))"
        
        return column
    
    def _render_scaling_expr(self, column: str, orig_column: str, method: str, params: dict) -> str:
        """Render SQL for scaling/normalization (Stage 2)."""
        normalized = self._normalize_method(method)
        
        if "standard" in normalized:
            # Use subquery for mean/std
            return f"({column} - (SELECT AVG({orig_column}) FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`)) / NULLIF((SELECT STDDEV({orig_column}) FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`), 0)"
        elif "normal" in normalized or "min_max" in normalized:
            return f"({column} - (SELECT MIN({orig_column}) FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`)) / NULLIF((SELECT MAX({orig_column}) - MIN({orig_column}) FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`), 0)"
        elif "log" in normalized:
            return f"LOG({column} + 1)"
        elif "sqrt" in normalized:
            return f"SQRT(ABS({column}))"
        
        return column
    
    def generate(
        self,
        transformations: List[Dict[str, Any]],
        target_column: Optional[str] = None,
        include_target: bool = True
    ) -> Dict[str, Any]:
        """
        Generate Airflow DAG code for ML-ready transformed data.
        Uses CTE approach to avoid nested analytic functions.
        """
        validated_transformations = []
        skipped = []
        
        for t in transformations:
            column = t.get("column_name", "")
            method = t.get("fe_method", "")
            
            if not column or not method:
                skipped.append({"column": column, "reason": "Missing column_name or fe_method"})
                continue
            
            if not validate_column_name(column):
                sanitized = sanitize_column_name(column)
                t["column_name"] = sanitized
            
            validated_transformations.append(t)
        
        if not validated_transformations:
            raise ValueError("No valid transformations provided")
        
        grouped = self._group_transformations_by_column(validated_transformations)
        
        # Stage 1: Build imputation expressions
        impute_selects = []
        imputed_columns = set()  # Track which columns have imputation
        
        for column, transforms in grouped.items():
            if column == target_column:
                continue
                
            for t in transforms:
                method = t.get("fe_method", "")
                params = {k: v for k, v in t.items() if k not in ("column_name", "fe_method")}
                
                if self._is_imputation(method):
                    expr = self._render_imputation_expr(column, method, params)
                    impute_selects.append(f"{expr} AS {column}_imputed")
                    imputed_columns.add(column)
                    break  # Only one imputation per column
        
        # Add pass-through for columns without imputation
        all_transform_columns = set(grouped.keys()) - {target_column} if target_column else set(grouped.keys())
        for column in all_transform_columns:
            if column not in imputed_columns:
                impute_selects.append(f"{column} AS {column}_imputed")
        
        # Add target column pass-through
        if target_column:
            impute_selects.append(f"{target_column} AS {target_column}_raw")
        
        # Stage 2: Build encoding/scaling expressions on imputed columns
        final_selects = []
        column_metadata = []
        
        for column, transforms in grouped.items():
            if column == target_column:
                continue
            
            imputed_col = f"{column}_imputed"
            output_name = column
            applied_transforms = []
            final_expr = imputed_col
            
            for t in transforms:
                method = t.get("fe_method", "")
                params = {k: v for k, v in t.items() if k not in ("column_name", "fe_method")}
                
                if self._is_imputation(method):
                    applied_transforms.append("imputed")
                elif self._is_encoding(method):
                    final_expr = self._render_encoding_expr(imputed_col, method, params)
                    applied_transforms.append("encoded")
                elif self._is_scaling(method):
                    final_expr = self._render_scaling_expr(imputed_col, column, method, params)
                    applied_transforms.append("scaled")
                elif "log" in self._normalize_method(method):
                    final_expr = f"LOG({imputed_col} + 1)"
                    applied_transforms.append("log")
                elif "sqrt" in self._normalize_method(method):
                    final_expr = f"SQRT(ABS({imputed_col}))"
                    applied_transforms.append("sqrt")
            
            # Build output column name
            if applied_transforms:
                unique_suffixes = []
                for s in applied_transforms:
                    if s not in unique_suffixes:
                        unique_suffixes.append(s)
                output_name = f"{column}_{'_'.join(unique_suffixes)}"
            
            final_selects.append(f"{final_expr} AS {output_name}")
            column_metadata.append({
                "original_column": column,
                "output_column": output_name,
                "transformations": [t.get("fe_method") for t in transforms]
            })
        
        # Handle target column
        if target_column and include_target:
            target_expr = f"CASE WHEN CAST({target_column}_raw AS STRING) IN ('1', 'Yes', 'yes', 'YES', 'true', 'True', 'TRUE') THEN 1 ELSE 0 END"
            final_selects.append(f"{target_expr} AS target")
            column_metadata.append({
                "original_column": target_column,
                "output_column": "target",
                "transformations": ["auto_binary_encoding"],
                "note": "Target auto-encoded to 0/1 for ML-ready output"
            })
        
        # Build CTE-based SQL
        target_full = f"`{self.project_id}.{self.target_dataset}.{self.target_table}`"
        source_full = f"`{self.project_id}.{self.dataset_id}.{self.source_table}`"
        
        impute_clause = ",\n    ".join(impute_selects)
        final_clause = ",\n    ".join(final_selects)
        
        sql_query = f"""
CREATE OR REPLACE TABLE {target_full} AS
WITH imputed AS (
    SELECT
    {impute_clause}
    FROM {source_full}
)
SELECT
    {final_clause}
FROM imputed
"""
        
        task_def = self.TASK_TEMPLATE.format(
            task_id="apply_ml_transformations",
            description="Apply all ML-ready transformations (CTE-based: impute → encode)",
            sql_query=sql_query
        )
        
        task_deps = "    # Single task - no dependencies needed"
        
        dag_code = self.DAG_TEMPLATE.format(
            generated_at=datetime.utcnow().isoformat(),
            project_id=self.project_id,
            source_dataset=self.dataset_id,
            source_table=self.source_table,
            target_dataset=self.target_dataset,
            target_table=self.target_table,
            dag_id=self.dag_id,
            task_definitions=task_def,
            task_dependencies=task_deps
        )
        
        return {
            "status": "success",
            "dag_code": dag_code,
            "dag_code_lines": dag_code.split('\n'),
            "dag_id": self.dag_id,
            "dag_name": self.dag_id,
            "target_table_name": self.target_table,
            "target_dataset": self.target_dataset,
            "source_table": self.source_table,
            "target_column": target_column,
            "transformations_applied": validated_transformations,
            "transformations_skipped": skipped,
            "column_mapping": column_metadata,
            "generation_method": "cte_chained",
        }
    
    def get_dag_id(self) -> str:
        """Return the generated DAG ID."""
        return self.dag_id
    
    def preview_sql(
        self, 
        transformations: List[Dict[str, Any]],
        target_column: Optional[str] = None
    ) -> str:
        """Preview the SQL that would be generated."""
        import re
        result = self.generate(transformations, target_column)
        match = re.search(r'"""(CREATE OR REPLACE.*?)"""', result["dag_code"], re.DOTALL)
        return match.group(1).strip() if match else "SQL extraction failed"
