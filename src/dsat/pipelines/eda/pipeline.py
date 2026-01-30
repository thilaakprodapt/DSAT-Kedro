"""EDA Pipeline Definition.

Defines the Kedro pipeline for Exploratory Data Analysis.
"""

from kedro.pipeline import Pipeline, node, pipeline

from dsat.pipelines.eda.nodes import (
    load_data_from_bq,
    detect_column_types,
    compute_data_overview,
    compute_missing_values,
    compute_univariate_numerical,
    compute_univariate_categorical,
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
            name="detect_types_node",
        ),
        node(
            func=compute_data_overview,
            inputs="eda_raw_data",
            outputs="data_overview",
            name="compute_overview_node",
        ),
        node(
            func=compute_missing_values,
            inputs="eda_raw_data",
            outputs="missing_values",
            name="compute_missing_node",
        ),
        node(
            func=compute_univariate_numerical,
            inputs={
                "df": "eda_raw_data",
                "column_types": "column_types",
            },
            outputs="univariate_numerical",
            name="compute_univariate_numerical_node",
        ),
        node(
            func=compute_univariate_categorical,
            inputs={
                "df": "eda_raw_data",
                "column_types": "column_types",
            },
            outputs="univariate_categorical",
            name="compute_univariate_categorical_node",
        ),
        node(
            func=generate_eda_summary,
            inputs={
                "data_overview": "data_overview",
                "missing_values": "missing_values",
                "univariate_numerical": "univariate_numerical",
                "univariate_categorical": "univariate_categorical",
                "column_types": "column_types",
            },
            outputs="eda_summary",
            name="generate_summary_node",
        ),
    ])
