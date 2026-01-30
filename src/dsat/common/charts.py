"""Chart generation utilities for DSAT.

Creates visualizations and uploads them to GCS.
"""

import matplotlib
matplotlib.use('Agg')

import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)

# Chart styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.autolayout"] = True


def upload_fig_to_gcs(
    fig: plt.Figure,
    name: str,
    client: gcs_storage.Client,
    bucket_name: str,
    folder: str = "charts-ds",
    expiration_hours: int = 24
) -> str:
    """Upload a matplotlib figure to GCS and return a signed URL.
    
    Args:
        fig: matplotlib figure object
        name: base name for the file
        client: GCS client
        bucket_name: GCS bucket name
        folder: folder within bucket
        expiration_hours: how long the signed URL should be valid
    
    Returns:
        Signed URL for the uploaded image
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{ts}.png"
    blob_path = f"{folder}/{filename}"

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)

    blob.upload_from_string(buf.getvalue(), content_type='image/png')
    buf.close()

    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=expiration_hours),
        method="GET"
    )
    
    logger.info(f"Uploaded chart: gs://{bucket_name}/{blob_path}")
    return signed_url


def univariate_numerical(
    df: pd.DataFrame,
    num_cols: List[str],
    client: Optional[gcs_storage.Client] = None,
    bucket_name: Optional[str] = None
) -> Optional[str]:
    """Create histogram grid for numerical columns.
    
    Args:
        df: Input DataFrame
        num_cols: List of numerical column names
        client: Optional GCS client for upload
        bucket_name: Optional GCS bucket name
    
    Returns:
        Signed URL if uploaded, else None
    """
    if not num_cols:
        return None

    cols = 2
    rows = (len(num_cols) + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(col)

    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Histogram Grid (Numerical)", fontsize=16)
    
    url = None
    if client and bucket_name:
        url = upload_fig_to_gcs(fig, "univariate_numerical", client, bucket_name)
    
    plt.close(fig)
    return url


def univariate_categorical(
    df: pd.DataFrame,
    cat_cols: List[str],
    client: Optional[gcs_storage.Client] = None,
    bucket_name: Optional[str] = None
) -> Optional[str]:
    """Create bar chart grid for categorical columns.
    
    Args:
        df: Input DataFrame
        cat_cols: List of categorical column names
        client: Optional GCS client for upload
        bucket_name: Optional GCS bucket name
    
    Returns:
        Signed URL if uploaded, else None
    """
    if not cat_cols:
        return None

    cols = 2
    rows = (len(cat_cols) + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, col in enumerate(cat_cols):
        if i >= len(axes):
            break
            
        clean_series = df[col].astype(str).str.strip().replace("", np.nan).dropna()

        if clean_series.empty:
            axes[i].text(0.5, 0.5, f"No valid data\nin {col}",
                         ha='center', va='center', fontsize=12)
            axes[i].set_title(col)
            axes[i].set_axis_off()
            continue

        top_values = clean_series.value_counts().head(10).index
        sns.countplot(x=clean_series, order=top_values, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Bar Chart Grid (Categorical)", fontsize=16)
    
    url = None
    if client and bucket_name:
        url = upload_fig_to_gcs(fig, "univariate_categorical", client, bucket_name)
    
    plt.close(fig)
    return url


def numeric_target_analysis(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    client: Optional[gcs_storage.Client] = None,
    bucket_name: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """Create visualizations for numeric target analysis.
    
    Args:
        df: Input DataFrame
        target: Target column name
        num_cols: Numerical column names
        cat_cols: Categorical column names
        client: Optional GCS client
        bucket_name: Optional bucket name
    
    Returns:
        Dict of chart name to signed URL
    """
    results = {}

    # Correlation Bar Chart
    numeric_features = [c for c in num_cols if c != target]
    if numeric_features:
        corr = df[numeric_features].corrwith(df[target]).sort_values()

        fig, ax = plt.subplots(figsize=(10, 7))
        corr.plot(kind='bar', ax=ax)
        ax.set_title(f"Correlation with Target: {target}")
        ax.set_ylabel("Correlation")

        if client and bucket_name:
            results["correlation_bar"] = upload_fig_to_gcs(
                fig, "target_numeric_correlation", client, bucket_name
            )
        plt.close(fig)

    # Categorical vs Target - Mean Plot Grid
    if cat_cols:
        cols = 2
        rows = (len(cat_cols) + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        for i, col in enumerate(cat_cols):
            if i >= len(axes):
                break
            if df[col].nunique() > 20:
                df[col] = df[col].astype(str)

            sns.barplot(x=col, y=target, data=df, ax=axes[i], estimator=np.mean)
            axes[i].set_title(f"{col} vs {target} (Mean)")
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(cat_cols), len(axes)):
            axes[j].set_visible(False)

        if client and bucket_name:
            results["cat_mean_grid"] = upload_fig_to_gcs(
                fig, "target_cat_mean_grid", client, bucket_name
            )
        plt.close(fig)

    return results


def categorical_target_analysis(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    client: Optional[gcs_storage.Client] = None,
    bucket_name: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """Create visualizations for categorical target analysis.
    
    Args:
        df: Input DataFrame
        target: Target column name
        num_cols: Numerical column names
        cat_cols: Categorical column names
        client: Optional GCS client
        bucket_name: Optional bucket name
    
    Returns:
        Dict of chart name to signed URL
    """
    results = {}

    # Numerical vs Target - Boxplot Grid
    if num_cols:
        cols = 2
        rows = (len(num_cols) + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        for i, num in enumerate(num_cols):
            if i >= len(axes):
                break
            sns.boxplot(x=df[target], y=df[num], ax=axes[i])
            axes[i].set_title(f"{num} vs {target}")
            axes[i].tick_params(axis='x', rotation=25)

        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)

        if client and bucket_name:
            results["boxplot_grid"] = upload_fig_to_gcs(
                fig, "target_boxplot_grid", client, bucket_name
            )
        plt.close(fig)

    # Chi-Square Importance
    scores = {}
    for col in cat_cols:
        try:
            confusion = pd.crosstab(df[col], df[target])
            chi2 = stats.chi2_contingency(confusion)[0]
            scores[col] = chi2
        except Exception:
            scores[col] = 0

    if scores:
        scores_series = pd.Series(scores).sort_values()

        fig, ax = plt.subplots(figsize=(10, 7))
        scores_series.plot(kind="barh", ax=ax)
        ax.set_title(f"Categorical Importance (Chi-square with {target})")
        ax.set_xlabel("Chi-square Score")

        if client and bucket_name:
            results["chi_square_importance"] = upload_fig_to_gcs(
                fig, "target_chi_square_importance", client, bucket_name
            )
        plt.close(fig)

    return results
