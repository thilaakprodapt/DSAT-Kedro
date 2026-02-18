from google.cloud import bigquery
from datetime import datetime, timezone
from random import getrandbits
import json

# These should already exist somewhere in your project
# from dsat.config import WORKSPACE_details, LEAK_TABLE_ID

WORKSPACE_details = "cloud-practice-dev-2.DS_details.Workspace_details"
LEAK_TABLE_ID = "cloud-practice-dev-2.DS_details.Leakage_Detection_details"
def set_phase_status(
    bq_client: bigquery.Client,
    table_name: str,
    phase_col: str,
    status: str
):
    sql = f"""
    UPDATE `{WORKSPACE_details}`
    SET {phase_col} = @status,
        Last_updated = CURRENT_DATETIME()
    WHERE table_name = @table_name
    """

    job = bq_client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("status", "STRING", status),
                bigquery.ScalarQueryParameter("table_name", "STRING", table_name),
            ]
        ),
    )

    job.result()


def save_leakage_to_bq(
    bq_client: bigquery.Client,
    dataset: str,
    table: str,
    data: dict,
    full_output: str
):
    """
    Saves leakage analysis to BigQuery using batch load
    """

    summary = data.get("summary", {}) or {}

    def norm_list(key):
        arr = data.get(key, []) or []
        return [
            {
                "feature": x.get("feature", ""),
                "reason": x.get("reason", "")
            }
            for x in arr if isinstance(x, dict)
        ]

    row = {
        "id": int(getrandbits(63)),
        "dataset": dataset,
        "table_name": table,
        "summary": {
            "high_risk": int(summary.get("high_risk", 0) or 0),
            "medium_risk": int(summary.get("medium_risk", 0) or 0),
            "low_risk": int(summary.get("low_risk", 0) or 0),
        },
        "high_risk_features": norm_list("high_risk_features"),
        "medium_risk_features": norm_list("medium_risk_features"),
        "low_risk_features": norm_list("low_risk_features"),
        "ld_full_output": full_output,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,
    )

    load_job = bq_client.load_table_from_json(
        json_rows=[row],
        destination=LEAK_TABLE_ID,
        job_config=job_config
    )

    load_job.result()
    print(f"Leakage batch job {load_job.job_id} completed")

    return row["id"]
