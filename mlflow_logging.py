import mlflow
from typing import Dict

def log_mlflow_run(run_name: str, model_type: str, metrics: Dict[str, float]):
    mlflow.set_tracking_uri("file:/content/mlruns")
    mlflow.set_experiment("fraud_detection_experiments")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_type)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        print(f"✅ Logged run: {run_name}")
