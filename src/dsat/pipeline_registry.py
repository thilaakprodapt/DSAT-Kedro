"""Pipeline registry for DSAT - Kedro 1.x."""

from kedro.pipeline import Pipeline

from dsat.pipelines.eda import create_pipeline as create_eda_pipeline
from dsat.pipelines.feature_engineering import create_pipeline as create_fe_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.
    
    Returns:
        A dictionary mapping pipeline names to Pipeline objects.
    
    Usage:
        kedro run                           # Run default (EDA) pipeline
        kedro run --pipeline=eda            # Run EDA pipeline
        kedro run --pipeline=fe             # Run Feature Engineering pipeline
        kedro run --pipeline=data_science   # Run EDA + FE pipeline
    """
    eda_pipeline = create_eda_pipeline()
    fe_pipeline = create_fe_pipeline()
    
    return {
        "eda": eda_pipeline,
        "fe": fe_pipeline,
        "feature_engineering": fe_pipeline,
        "data_science": eda_pipeline + fe_pipeline,
        "__default__": eda_pipeline,
    }
