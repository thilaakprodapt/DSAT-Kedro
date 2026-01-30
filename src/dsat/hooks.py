"""Project hooks for DSAT - Kedro 1.x compatible."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ProjectHooks:
    """Basic project hooks - MLFlow is handled by kedro-mlflow plugin."""

    def after_catalog_created(self, catalog) -> None:
        """Log catalog creation."""
        logger.info(f"Catalog created with {len(catalog.list())} datasets")

    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Log pipeline start."""
        pipeline_name = run_params.get("pipeline_name", "__default__")
        logger.info(f"Starting pipeline: {pipeline_name}")

    def after_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Log pipeline completion."""
        pipeline_name = run_params.get("pipeline_name", "__default__")
        logger.info(f"Completed pipeline: {pipeline_name}")

    def on_pipeline_error(self, error: Exception, run_params: Dict[str, Any]) -> None:
        """Log pipeline errors."""
        pipeline_name = run_params.get("pipeline_name", "__default__")
        logger.error(f"Pipeline {pipeline_name} failed: {error}")
