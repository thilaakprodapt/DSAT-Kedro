"""DAG Trigger utilities for Airflow integration."""

import subprocess
import time
import logging

logger = logging.getLogger(__name__)

SCHEDULER = "airflow-scheduler-1"  # Docker container name


def run_cmd(cmd: str) -> tuple:
    """Run shell commands and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()


def dag_exists(dag_id: str) -> bool:
    """Check if DAG exists in Airflow."""
    out, _ = run_cmd(f"docker exec {SCHEDULER} airflow dags list")
    return dag_id in out


def is_dag_paused(dag_id: str) -> bool:
    """Check if DAG is paused."""
    out, _ = run_cmd(f"docker exec {SCHEDULER} airflow dags list --output table")
    for line in out.split("\n"):
        if dag_id in line:
            return "True" in line or "paused" in line.lower()
    return False


def unpause_dag(dag_id: str) -> None:
    """Unpause the DAG."""
    logger.info(f"Unpausing DAG: {dag_id}")
    run_cmd(f"docker exec {SCHEDULER} airflow dags unpause {dag_id}")


def trigger_dag(dag_id: str) -> None:
    """Trigger the DAG."""
    logger.info(f"Triggering DAG: {dag_id}")
    run_cmd(f"docker exec {SCHEDULER} airflow dags trigger {dag_id}")


def wait_for_dag_and_trigger(dag_id: str, timeout_seconds: int = 120) -> bool:
    """Wait for DAG → unpause if needed → trigger.
    
    Args:
        dag_id: The DAG ID to wait for and trigger
        timeout_seconds: Maximum time to wait for DAG to appear
    
    Returns:
        True if DAG was triggered successfully, False otherwise
    """
    logger.info(f"Checking for DAG '{dag_id}'...")

    # Wait for DAG to appear
    wait_interval = 3
    max_attempts = timeout_seconds // wait_interval
    
    for _ in range(max_attempts):
        if dag_exists(dag_id):
            logger.info(f"✅ DAG '{dag_id}' detected in Airflow!")
            break
        logger.info("DAG not found. Waiting...")
        time.sleep(wait_interval)
    else:
        logger.error("⛔ Timeout: DAG not detected.")
        return False

    # Check paused state
    if is_dag_paused(dag_id):
        logger.info(f"⚠️ DAG '{dag_id}' is paused.")
        unpause_dag(dag_id)
    else:
        logger.info("✔ DAG already unpaused.")

    # Trigger DAG
    trigger_dag(dag_id)
    logger.info("🎉 DAG triggered successfully!")
    return True
