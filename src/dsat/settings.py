"""Project settings for Kedro 1.x.

Documentation: https://docs.kedro.org/en/stable/kedro_project_setup/settings.html
"""

from dsat.hooks import ProjectHooks

# Instantiate and list hooks
HOOKS = (ProjectHooks(),)

# Disable telemetry
TELEMETRY = False
