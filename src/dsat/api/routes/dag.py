"""DAG Generation API routes - matching original endpoints exactly."""

import os
import subprocess
import time
import logging
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

router = APIRouter()

# Airflow scheduler container name (same as original)
SCHEDULER = "airflow-scheduler-1"


class Transformation(BaseModel):
    column_name: str
    fe_method: str


class DAGRequest(BaseModel):
    project_id: str
    dataset_id: str
    source_table: str
    target_dataset: str
    target_column: Optional[str] = None
    transformation: List[Transformation]


class SaveDAGRequest(BaseModel):
    dag_code: str
    dag_name: str
    target_table_name: str
    target_dataset: str
    source_table: str
    input_data: dict


# =============================================================================
# DAG TRIGGER HELPER FUNCTIONS (inline like original)
# =============================================================================

def run_cmd(cmd):
    """Run shell commands and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()


def dag_exists(dag_id):
    """Check if DAG exists in Airflow."""
    out, _ = run_cmd(f"docker exec {SCHEDULER} airflow dags list")
    return dag_id in out


def is_dag_paused(dag_id):
    """Check if DAG is paused."""
    out, _ = run_cmd(f"docker exec {SCHEDULER} airflow dags list --output table")
    for line in out.split("\n"):
        if dag_id in line:
            return "True" in line or "paused" in line.lower()
    return False


def unpause_dag(dag_id):
    """Unpause the DAG."""
    print(f"Unpausing DAG: {dag_id}")
    run_cmd(f"docker exec {SCHEDULER} airflow dags unpause {dag_id}")


def trigger_dag(dag_id):
    """Trigger the DAG."""
    print(f"Triggering DAG: {dag_id}")
    run_cmd(f"docker exec {SCHEDULER} airflow dags trigger {dag_id}")


def wait_for_dag_and_trigger(dag_id):
    """Wait for DAG → unpause if needed → trigger."""
    print(f"Checking for DAG '{dag_id}'...")

    # Wait up to 2 minutes
    for _ in range(40):
        if dag_exists(dag_id):
            print(f"✅ DAG '{dag_id}' detected in Airflow!")
            break
        print("DAG not found. Waiting...")
        time.sleep(3)
    else:
        print("⛔ Timeout: DAG not detected.")
        return False

    # Check paused state
    if is_dag_paused(dag_id):
        print(f"⚠️ DAG '{dag_id}' is paused.")
        unpause_dag(dag_id)
    else:
        print("✔ DAG already unpaused.")

    # Trigger DAG
    trigger_dag(dag_id)
    print("🎉 DAG triggered successfully!")
    return True


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/generate_dag")
async def generate_dag(input: dict) -> Dict[str, Any]:
    """Generate Airflow DAG code from transformations."""
    try:
        from dsat.common import DAGGenerator
        
        project_id = input.get("project_id")
        dataset_id = input.get("dataset_id")
        source_table = input.get("source_table")
        target_dataset = input.get("target_dataset")
        transformations = input.get("transformation", [])
        target_column = input.get("target_column")
        
        if not all([project_id, dataset_id, source_table, target_dataset, transformations]):
            raise ValueError("Missing required fields")
        
        generator = DAGGenerator(
            project_id=project_id,
            dataset_id=dataset_id,
            source_table=source_table,
            target_dataset=target_dataset
        )
        
        result = generator.generate(
            transformations=transformations,
            target_column=target_column
        )
        
        result["input_data"] = input
        result["generation_method"] = "template"
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating DAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save_dag")
def save_dag(request: SaveDAGRequest) -> Dict[str, Any]:
    """Save DAG to Airflow dags folder and trigger it.
    
    EXACTLY like original implementation.
    """
    dag_name = request.dag_name
    dag_code = request.dag_code
    target_table_name = request.target_table_name
    target_dataset = request.target_dataset
    source_table = request.source_table

    try:
        # Save DAG file - EXACT same path as original
        file_path = f"/home/airflow/dags/{dag_name}.py"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(dag_code)
        print(f"DAG saved at {file_path}")

        # Trigger DAG via Docker - EXACT same as original
        triggered = wait_for_dag_and_trigger(dag_name)
        if not triggered:
            print(f"⚠️ Failed to trigger DAG {dag_name}")
            return {
                "status": "error",
                "message": f"Failed to trigger DAG {dag_name}"
            }

        return {
            "status": "success",
            "dag_name": dag_name,
            "target_table_name": target_table_name,
            "target_dataset": target_dataset,
            "file_path": file_path,
            "message": f"DAG created and triggered successfully"
        }

    except Exception as e:
        print(f"Error during DAG save/trigger: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/preview_sql")
async def preview_sql(input: dict) -> Dict[str, Any]:
    """Preview the SQL that would be generated."""
    try:
        from dsat.common import DAGGenerator
        
        generator = DAGGenerator(
            project_id=input.get("project_id"),
            dataset_id=input.get("dataset_id"),
            source_table=input.get("source_table"),
            target_dataset=input.get("target_dataset")
        )
        
        sql = generator.preview_sql(
            transformations=input.get("transformation", []),
            target_column=input.get("target_column")
        )
        
        return {"status": "success", "sql": sql}
        
    except Exception as e:
        logger.error(f"Error previewing SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fe_chat_check")
def fe_chat_check(input: str) -> Dict[str, Any]:
    """Check if input is a valid FE technique."""
    valid_techniques = {
        "standardization", "normalization", "log_transformation",
        "label_encoding", "frequency_encoding", "target_encoding",
        "impute_mean", "impute_median", "impute_mode",
    }
    
    input_lower = input.lower().strip()
    is_valid = input_lower in valid_techniques
    
    suggestion = ""
    if not is_valid:
        if "encod" in input_lower:
            suggestion = "Try: label encoding, target encoding"
        elif "imput" in input_lower:
            suggestion = "Try: impute_mean, impute_median, impute_mode"
        else:
            suggestion = "Check that input is a valid ML technique"
    
    return {"valid": is_valid, "suggestion": suggestion}


@router.get("/available_transformations")
async def available_transformations() -> Dict[str, Any]:
    return {
        "status": "success",
        "categories": {
            "numerical": ["standardization", "normalization", "log_transformation"],
            "categorical": ["label_encoding", "frequency_encoding"],
            "missing_values": ["impute_mean", "impute_median", "impute_mode"],
        }
    }
