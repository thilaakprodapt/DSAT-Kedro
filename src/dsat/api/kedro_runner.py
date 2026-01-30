"""Kedro Session Runner for API integration.

Provides a clean interface to run Kedro pipelines from FastAPI endpoints.
Compatible with Kedro 0.18.x and 1.x.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_project_path() -> Path:
    """Get the Kedro project root path."""
    current = Path(__file__).resolve()
    # Go up: kedro_runner.py -> api -> dsat -> src -> DSAT
    return current.parent.parent.parent.parent


@contextmanager
def get_kedro_session(
    pipeline_name: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None
):
    """Context manager for Kedro session.
    
    Compatible with both Kedro 0.18.x and 1.x.
    """
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    import inspect
    
    project_path = get_project_path()
    bootstrap_project(project_path)
    
    # Check Kedro version by inspecting KedroSession.create signature
    create_sig = inspect.signature(KedroSession.create)
    params = create_sig.parameters
    
    # Build kwargs based on what's supported
    create_kwargs = {"project_path": project_path}
    
    if "extra_params" in params:
        create_kwargs["extra_params"] = extra_params or {}
    
    with KedroSession.create(**create_kwargs) as session:
        yield session


def run_pipeline(
    pipeline_name: str,
    extra_params: Optional[Dict[str, Any]] = None,
    node_names: Optional[list] = None,
    from_nodes: Optional[list] = None,
    to_nodes: Optional[list] = None
) -> Dict[str, Any]:
    """Run a Kedro pipeline and return outputs."""
    with get_kedro_session(pipeline_name, extra_params) as session:
        outputs = session.run(
            pipeline_name=pipeline_name,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes
        )
        return outputs or {}


def get_catalog():
    """Get the Kedro data catalog."""
    with get_kedro_session() as session:
        context = session.load_context()
        return context.catalog


def get_pipelines() -> List[str]:
    """Get all registered pipeline names.
    
    Compatible with different Kedro versions.
    """
    try:
        # Try importing from pipeline_registry directly
        from dsat.pipeline_registry import register_pipelines
        pipelines = register_pipelines()
        return list(pipelines.keys())
    except Exception as e:
        logger.warning(f"Could not get pipelines from registry: {e}")
        # Fallback: return known pipelines
        return ["eda", "fe", "feature_engineering", "data_science", "__default__"]
