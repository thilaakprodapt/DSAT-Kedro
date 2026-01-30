"""Kedro Session Runner for API integration.

Provides a clean interface to run Kedro pipelines from FastAPI endpoints
with proper session management, MLFlow tracking, and data catalog support.
Compatible with Kedro 0.18.x and 1.x.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_project_path() -> Path:
    """Get the Kedro project root path."""
    # Navigate from src/dsat/api to project root
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
    
    Args:
        pipeline_name: Optional pipeline to run
        extra_params: Optional runtime parameters
    
    Yields:
        KedroSession instance
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
    """Run a Kedro pipeline and return outputs.
    
    Args:
        pipeline_name: Name of the pipeline to run
        extra_params: Runtime parameters override
        node_names: Specific nodes to run
        from_nodes: Start from these nodes
        to_nodes: Run up to these nodes
    
    Returns:
        Dict of output dataset names to values
    """
    with get_kedro_session(pipeline_name, extra_params) as session:
        # Run the pipeline
        outputs = session.run(
            pipeline_name=pipeline_name,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes
        )
        
        return outputs or {}


def get_catalog():
    """Get the Kedro data catalog.
    
    Returns:
        DataCatalog instance
    """
    with get_kedro_session() as session:
        context = session.load_context()
        return context.catalog


def get_pipelines() -> Dict[str, Any]:
    """Get all registered pipelines.
    
    Returns:
        Dict of pipeline names to Pipeline objects
    """
    with get_kedro_session() as session:
        context = session.load_context()
        return context.pipelines
