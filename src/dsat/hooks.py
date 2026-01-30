"""Project hooks for DSAT - Kedro 1.x with MLFlow integration."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ProjectHooks:
    """Project hooks with MLFlow tracking."""

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


class MLFlowHooks:
    """MLFlow tracking hooks for experiment management."""

    def __init__(self):
        self._mlflow = None

    @property
    def mlflow(self):
        """Lazy load mlflow."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLFlow not installed, tracking disabled")
                self._mlflow = None
        return self._mlflow

    def before_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """Start MLFlow run before pipeline execution."""
        if not self.mlflow:
            return
            
        pipeline_name = run_params.get("pipeline_name", "__default__")
        
        try:
            # Set experiment
            self.mlflow.set_experiment("DSAT_Experiments")
            
            # Start run
            self.mlflow.start_run(run_name=f"pipeline_{pipeline_name}")
            
            # Log parameters
            self.mlflow.log_param("pipeline_name", pipeline_name)
            
            logger.info(f"MLFlow run started for pipeline: {pipeline_name}")
        except Exception as e:
            logger.warning(f"Failed to start MLFlow run: {e}")

    def after_node_run(
        self,
        node,
        catalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Log node outputs to MLFlow."""
        if not self.mlflow:
            return
            
        try:
            # Log node completion
            self.mlflow.log_param(f"node_{node.name}_completed", True)
            
            # Log metrics if outputs contain numeric values
            for name, value in outputs.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(name, value)
                elif isinstance(value, dict):
                    # Log nested metrics
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            self.mlflow.log_metric(f"{name}_{k}", v)
        except Exception as e:
            logger.debug(f"Failed to log node outputs: {e}")

    def after_pipeline_run(self, run_params: Dict[str, Any]) -> None:
        """End MLFlow run after pipeline completion."""
        if not self.mlflow:
            return
            
        try:
            self.mlflow.end_run()
            logger.info("MLFlow run completed")
        except Exception as e:
            logger.warning(f"Failed to end MLFlow run: {e}")

    def on_pipeline_error(self, error: Exception, run_params: Dict[str, Any]) -> None:
        """Handle pipeline errors in MLFlow."""
        if not self.mlflow:
            return
            
        try:
            self.mlflow.log_param("error", str(error)[:250])
            self.mlflow.end_run(status="FAILED")
        except Exception as e:
            logger.warning(f"Failed to log error to MLFlow: {e}")
