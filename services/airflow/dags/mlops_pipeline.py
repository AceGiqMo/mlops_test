from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "acegiqmo",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(seconds=15),
}


with DAG(
    "wine_mlops_pipeline",
    default_args=default_args,
    description="Automated pipeline of preprocessing, training and deploy of ML model",
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "wine", "mlops"],
) as dag:

    # --- TASK 1: Установка зависимостей в общую папку ---
    install_deps = BashOperator(
        task_id="install_deps",
        bash_command="""
            set -e
            mkdir -p /tmp/mlops_deps
            python -m pip install -q --target /tmp/mlops_deps -r /opt/airflow/project/requirements.txt
            python -m pip install -q --target /tmp/mlops_deps pandas pyarrow
            echo "[SUCCESS] Dependencies installed to /tmp/mlops_deps"
        """,
    )

    # --- TASK 2: Preprocessing ---
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="""
            set -e
            cd /opt/airflow/project
            export PYTHONPATH=/tmp/mlops_deps
            python code/models/preprocess.py
        """,
    )

    # --- TASK 3: Training ---
    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
            set -e
            cd /opt/airflow/project
            export PYTHONPATH=/tmp/mlops_deps
            export MLFLOW_URI=http://mlflow:5000
            python code/models/train.py
        """,
    )

    # --- TASK 4: Evaluate & Reload API ---
    evaluate_and_reload_api = BashOperator(
        task_id="evaluate_and_reload_api",
        bash_command="""
            set -e
            export PYTHONPATH=/tmp/mlops_deps
            export MLFLOW_TRACKING_URI=http://mlflow:5000
            python - <<'PY'
import os
import sys

import requests
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

experiment = client.get_experiment_by_name("wine-classification")
if not experiment:
    sys.exit("[ERROR] Experiment 'wine-classification' not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attribute.start_time DESC"],
    max_results=1,
)
if not runs:
    sys.exit("[ERROR] No runs found")

latest_run = runs[0]
new_run_id = latest_run.info.run_id
new_accuracy = latest_run.data.metrics.get("test_accuracy", 0.0)
print(f"[INFO] New Run ID: {new_run_id}, Test Accuracy: {new_accuracy:.4f}")

model_versions = client.search_model_versions(f"run_id='{new_run_id}'")
if not model_versions:
    sys.exit("[ERROR] Model not registered in Registry")
new_version = int(model_versions[0].version)

all_versions = client.search_model_versions("name='wine-model'")
if len(all_versions) > 1:
    prev_version = sorted(int(v.version) for v in all_versions)[-2]
    prev_mv = client.get_model_version("wine-model", str(prev_version))
    prev_run = client.get_run(prev_mv.run_id)
    prev_accuracy = prev_run.data.metrics.get("test_accuracy", 0.0)
    print(f"[INFO] Previous version {prev_version} accuracy: {prev_accuracy:.4f}")
    if new_accuracy < prev_accuracy:
        print(f"[SKIPPED] New model ({new_accuracy:.4f}) worse than {prev_accuracy:.4f}")
        sys.exit(0)

print(f"[SUCCESS] Version {new_version} is best. Reloading API...")
resp = requests.post(
    "http://wine-api:8000/reload",
    headers={"X-API-Key": "secret"},
    params={"version": new_version},
)
print(f"[INFO] API response: {resp.status_code} {resp.text}")
resp.raise_for_status()
PY
        """,
    )

    install_deps >> preprocess_data >> train_model >> evaluate_and_reload_api