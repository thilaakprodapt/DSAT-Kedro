"""GCS Utilities for DSAT.

Utilities for working with Google Cloud Storage.
"""

import logging
from datetime import timedelta
from urllib.parse import urlparse
from typing import Any, Optional

from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)


def get_gcs_client(project_id: str, credentials=None) -> gcs_storage.Client:
    """Get a GCS client.
    
    Args:
        project_id: GCP project ID
        credentials: Optional credentials object
    
    Returns:
        GCS Client
    """
    return gcs_storage.Client(project=project_id, credentials=credentials)


def upload_to_gcs(
    client: gcs_storage.Client,
    bucket_name: str,
    blob_path: str,
    data: bytes,
    content_type: str = "image/png"
) -> str:
    """Upload data to GCS and return signed URL.
    
    Args:
        client: GCS client
        bucket_name: GCS bucket name
        blob_path: Path within the bucket
        data: Bytes to upload
        content_type: MIME type of the data
    
    Returns:
        Signed URL for the uploaded file
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    blob.upload_from_string(data, content_type=content_type)
    
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET"
    )
    
    logger.info(f"Uploaded to gs://{bucket_name}/{blob_path}")
    return signed_url


def generate_signed_url(
    client: gcs_storage.Client,
    bucket_name: str,
    blob_path: str,
    expiration_hours: int = 24
) -> Optional[str]:
    """Generate a signed URL for an existing GCS object.
    
    Args:
        client: GCS client
        bucket_name: GCS bucket name
        blob_path: Path within the bucket
        expiration_hours: URL validity in hours
    
    Returns:
        Signed URL or None if blob doesn't exist
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        logger.warning(f"Blob not found: gs://{bucket_name}/{blob_path}")
        return None
    
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=expiration_hours),
        method="GET"
    )


def refresh_signed_urls_in_data(
    data: Any, 
    client: gcs_storage.Client,
    expiration_hours: int = 24
) -> Any:
    """Recursively traverse data and regenerate signed URLs for GCS images.
    
    Args:
        data: Dictionary or list containing the analysis data
        client: GCS client
        expiration_hours: How long the new signed URLs should be valid
        
    Returns:
        Modified data with fresh signed URLs
    """
    
    def is_gcs_signed_url(url: str) -> bool:
        """Check if a URL is a GCS signed URL."""
        if not isinstance(url, str):
            return False
        return 'storage.googleapis.com' in url and 'X-Goog-Algorithm' in url
    
    def extract_gcs_path_from_url(url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract bucket and blob path from a signed URL."""
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) == 2:
                return path_parts[0], path_parts[1]
        except Exception:
            pass
        return None, None
    
    def regenerate_url(url: str) -> str:
        """Regenerate a signed URL."""
        try:
            bucket_name, blob_path = extract_gcs_path_from_url(url)
            if not bucket_name or not blob_path:
                return url
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"Blob not found: gs://{bucket_name}/{blob_path}")
                return url
            
            new_signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiration_hours),
                method="GET"
            )
            logger.debug(f"Refreshed URL for: {blob_path}")
            return new_signed_url
            
        except Exception as e:
            logger.warning(f"Failed to regenerate URL: {e}")
            return url
    
    def traverse(obj: Any) -> Any:
        """Recursively traverse and update URLs."""
        if isinstance(obj, dict):
            return {k: traverse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [traverse(item) for item in obj]
        elif isinstance(obj, str) and is_gcs_signed_url(obj):
            return regenerate_url(obj)
        else:
            return obj
    
    return traverse(data)
