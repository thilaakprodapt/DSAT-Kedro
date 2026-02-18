"""EDA Pipeline Definition.

Creates the Kedro pipeline for Exploratory Data Analysis.
Includes LLM-powered target detection and structured summary generation.
"""

from kedro.pipeline import Pipeline, node, pipeline

from dsat.pipelines.eda.nodes import (
    load_data_from_bq,
    detect_column_types,
    compute_data_overview,
    compute_missing_values,
    compute_outliers,
    compute_cardinality,
    compute_univariate_numerical,
    compute_univariate_categorical,
    find_target_column,
    generate_eda_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the EDA pipeline.

    Returns:
        Kedro Pipeline for EDA
    """
    return pipeline([
        node(
            func=load_data_from_bq,
            inputs={
                "project_id": "params:eda.table_config.project_id",
                "dataset_id": "params:eda.table_config.dataset_id",
                "table_name": "params:eda.table_config.table_name",
                "sample_limit": "params:eda.sample_limit",
            },
            outputs="eda_raw_data",
            name="load_data_node",
        ),
        node(
            func=detect_column_types,
            inputs="eda_raw_data",
            outputs="column_types",
            name="detect_column_types_node",
        ),
        node(
            func=compute_data_overview,
            inputs="eda_raw_data",
            outputs="data_overview",
            name="compute_data_overview_node",
        ),
        node(
            func=compute_missing_values,
            inputs="eda_raw_data",
            outputs="missing_values",
            name="compute_missing_values_node",
        ),
        node(
            func=compute_outliers,
            inputs={
                "df": "eda_raw_data",
                "column_types": "column_types",
            },
            outputs="outliers",
            name="compute_outliers_node",
        ),
        node(
            func=compute_cardinality,
            inputs="eda_raw_data",
            outputs="cardinality",
            name="compute_cardinality_node",
        ),
        node(
            func=compute_univariate_numerical,
            inputs={
                "df": "eda_raw_data",
                "column_types": "column_types",
            },
            outputs="univariate_numerical_stats",
            name="compute_univariate_numerical_node",
        ),
        node(
            func=compute_univariate_categorical,
            inputs={
                "df": "eda_raw_data",
                "column_types": "column_types",
            },
            outputs="univariate_categorical_stats",
            name="compute_univariate_categorical_node",
        ),
        node(
            func=find_target_column,
            inputs={
                "df": "eda_raw_data",
                "project_id": "params:gcp.project_id",
            },
            outputs="target_column",
            name="find_target_column_node",
        ),
        node(
            func=generate_eda_summary,
            inputs={
                "df": "eda_raw_data",
                "data_overview": "data_overview",
                "missing_values": "missing_values",
                "outliers": "outliers",
                "cardinality": "cardinality",
                "univariate_numerical": "univariate_numerical_stats",
                "univariate_categorical": "univariate_categorical_stats",
                "column_types": "column_types",
                "target_column": "target_column",
                "project_id": "params:gcp.project_id",
            },
            outputs="eda_summary",
            name="generate_eda_summary_node",
        ),
    ])
