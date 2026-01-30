# DSAT - Data Science Assistant Tool (Kedro)

A Kedro-based pipeline implementation of the Data Science Assistant Tool for automated ML workflows.

## Overview

This project provides modular, reproducible data science pipelines for:
- **EDA (Exploratory Data Analysis)** - Automated statistical analysis with Gemini AI
- **Feature Engineering** - AI-powered recommendations with SQL template transformations
- **DAG Generation** - Airflow DAG generation for BigQuery transformations
- **Data Balancing** - Handle imbalanced datasets
- **Leakage Detection** - Identify data leakage risks

## Project Structure

```
DSAT/
├── conf/                          # Configuration files
│   ├── base/                      # Default configuration
│   │   ├── catalog.yml            # Data catalog definitions
│   │   ├── parameters.yml         # Pipeline parameters
│   │   └── logging.yml            # Logging configuration
│   └── local/                     # Local overrides (gitignored)
│       └── credentials.yml        # GCP credentials
├── src/
│   └── dsat/
│       ├── pipelines/             # Kedro pipelines
│       │   ├── eda/               # EDA pipeline
│       │   ├── feature_engineering/
│       │   ├── dag_generation/
│       │   ├── data_balancing/
│       │   └── leakage_detection/
│       └── common/                # Shared utilities
├── tests/                         # Unit tests
└── pyproject.toml                 # Project configuration
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

## Usage

### Run a specific pipeline
```bash
kedro run --pipeline=eda
kedro run --pipeline=feature_engineering
```

### Run with parameters
```bash
kedro run --pipeline=eda --params="table_config.dataset=DS_FE_Dataset,table_config.table=Employee"
```

### Visualize pipelines
```bash
kedro viz
```

## Configuration

### Data Catalog (`conf/base/catalog.yml`)
Define your BigQuery tables and GCS artifacts here.

### Parameters (`conf/base/parameters.yml`)
Configure pipeline parameters like project ID, table names, etc.

### Credentials (`conf/local/credentials.yml`)
Add your GCP service account credentials (not tracked in git).

## Requirements

- Python 3.10+
- Google Cloud service account with BigQuery & GCS access
- Vertex AI API enabled

## License

Internal use only.
