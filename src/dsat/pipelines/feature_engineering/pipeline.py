"""Feature Engineering Pipeline Definition.

Creates the Kedro pipeline for Feature Engineering.
Uses LLM-powered recommendations from Gemini.
"""

from kedro.pipeline import Pipeline, node, pipeline

from dsat.pipelines.feature_engineering.nodes import (
    get_fe_recommendations,
    generate_dag_code,
    preview_sql,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Feature Engineering pipeline.

    Returns:
        Kedro Pipeline for Feature Engineering
    """
    return pipeline([
        node(
            func=get_fe_recommendations,
            inputs={
                "eda_summary": "eda_summary",
                "target_column": "params:fe.target_column",
                "project_id": "params:gcp.project_id",
            },
            outputs="fe_recommendations",
            name="get_recommendations_node",
        ),
        node(
            func=generate_dag_code,
            inputs={
                "recommendations": "fe_recommendations",
                "project_id": "params:gcp.project_id",
                "dataset_id": "params:fe.dataset_id",
                "table_name": "params:fe.table_name",
                "target_dataset": "params:fe.target_dataset",
                "target_column": "params:fe.target_column",
            },
            outputs="dag_result",
            name="generate_dag_node",
        ),
        node(
            func=preview_sql,
            inputs="dag_result",
            outputs="sql_preview",
            name="preview_sql_node",
        ),
    ])
