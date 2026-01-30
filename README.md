# DSAT - Data Science Assistant Tool

A Kedro-based data science pipeline for automated EDA, feature engineering, and DAG generation.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/thilaakprodapt/DSAT-Kedro.git
cd DSAT-Kedro
pip install -e .

# Run API server
python -m uvicorn dsat.api.main:app --reload --port 8001

# Open Swagger: http://localhost:8001/docs
```

## 📊 Features

| Feature | Description |
|---------|-------------|
| **EDA Pipeline** | Automated exploratory data analysis |
| **Feature Engineering** | Smart transformation recommendations |
| **DAG Generation** | Airflow DAG code from transformations |
| **Leakage Detection** | Identify data leakage risks |
| **MLFlow Tracking** | Experiment tracking for all runs |

## 🛠️ Kedro Commands

```bash
# List pipelines
kedro registry list

# Run EDA pipeline
kedro run --pipeline=eda

# Visualize pipelines
kedro viz

# View MLFlow experiments
mlflow ui --port 5000
```

## 📁 Project Structure

```
DSAT/
├── conf/                      # Configuration
│   ├── base/
│   │   ├── catalog.yml       # Data catalog
│   │   ├── parameters.yml    # Parameters
│   │   └── mlflow.yml        # MLFlow config
│   └── local/
│       └── credentials.yml   # GCP credentials
├── src/dsat/
│   ├── api/                  # FastAPI endpoints
│   ├── common/               # Shared utilities
│   ├── pipelines/            # Kedro pipelines
│   └── hooks.py              # MLFlow hooks
└── tests/                    # Unit tests
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/EDA/column_list` | GET | List table columns |
| `/EDA/analyze` | POST | Run EDA analysis |
| `/FeatureEngineering/recommendations` | POST | Get FE recommendations |
| `/Transformation/generate_dag` | POST | Generate Airflow DAG |
| `/LeakageDetection/detect` | POST | Detect data leakage |

## ⚙️ Configuration

Update `conf/base/parameters.yml` for your GCP project:

```yaml
gcp:
  project_id: your-project-id
  location: us-central1

eda:
  table_config:
    project_id: your-project-id
    dataset_id: your_dataset
    table_name: your_table
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📦 Dependencies

- Python >= 3.10
- Kedro >= 1.0.0
- FastAPI >= 0.100.0
- MLFlow >= 2.10.0
- Google Cloud BigQuery
- Google Cloud Storage

## 📝 License

MIT
