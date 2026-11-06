import os

import mlflow
import mlflow.keras


EXPERIMENT_NAME = "brain_tumor_models_docker_v3"


def setup_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_values(parameters=None, metrics=None, artifacts=None):
    for name, value in (parameters or {}).items():
        mlflow.log_param(name, value)
    for name, value in (metrics or {}).items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(name, value)
    for path in artifacts or []:
        if os.path.exists(path):
            mlflow.log_artifact(path)


def log_classification_run(model, history, metrics, parameters=None):
    with mlflow.start_run(run_name="classification"):
        values = {
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_precision": metrics["validation"]["precision"],
            "validation_recall": metrics["validation"]["recall"],
            "validation_f1_score": metrics["validation"]["f1_score"],
            "test_accuracy": metrics["test"]["accuracy"],
            "test_precision": metrics["test"]["precision"],
            "test_recall": metrics["test"]["recall"],
            "test_f1_score": metrics["test"]["f1_score"],
            "fine_tuned_test_accuracy": metrics["fine_tuned_test"]["accuracy"],
            "fine_tuned_test_f1_score": metrics["fine_tuned_test"]["f1_score"],
        }
        log_values(parameters, values)
        for name, values in history.history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(name, float(value), step=epoch)
        mlflow.keras.log_model(model, "classification_model")
        mlflow.log_dict(metrics, "classification_metrics.json")
        return mlflow.active_run().info.run_id


def log_detection_run(metrics, parameters=None, model_path=None):
    with mlflow.start_run(run_name="detection"):
        log_values(parameters, {
            "validation_precision": metrics["validation"]["precision"],
            "validation_recall": metrics["validation"]["recall"],
            "validation_map50": metrics["validation"]["map50"],
            "validation_map50_95": metrics["validation"]["map50_95"],
            "test_precision": metrics["test"]["precision"],
            "test_recall": metrics["test"]["recall"],
            "test_map50": metrics["test"]["map50"],
            "test_map50_95": metrics["test"]["map50_95"],
        })
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path)
        mlflow.log_dict(metrics, "detection_metrics.json")
        return mlflow.active_run().info.run_id


def run_classification_experiment(parameters=None):
    from src.classification import run_classification

    setup_mlflow()
    model, history, metrics = run_classification()
    run_id = log_classification_run(model, history, metrics, parameters)
    return run_id, metrics


def run_detection_experiment(parameters=None):
    from src.detection import run_detection

    setup_mlflow()
    metrics = run_detection()
    run_id = log_detection_run(
        metrics,
        parameters,
        model_path=metrics["model_path"],
    )
    return run_id, metrics


if __name__ == "__main__":
    setup_mlflow()
    print("MLflow experiment ready:", EXPERIMENT_NAME)
