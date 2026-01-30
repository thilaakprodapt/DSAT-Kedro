"""Project hooks for DSAT - Kedro 1.x compatible."""

import logging

logger = logging.getLogger(__name__)


class ProjectHooks:
    """Basic project hooks."""

    def after_catalog_created(self, catalog):
        """Log catalog creation."""
        logger.info(f"Catalog created with {len(catalog.list())} datasets")
