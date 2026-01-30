"""Project settings for Kedro 1.x with MLFlow.

Documentation: https://docs.kedro.org/en/stable/kedro_project_setup/settings.html
"""

from dsat.hooks import ProjectHooks, MLFlowHooks

# Instantiate and list hooks
HOOKS = (ProjectHooks(), MLFlowHooks())

# Disable telemetry
TELEMETRY = False
