import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_DIR = Path(
    os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[1])
)
sys.path.insert(0, str(PROJECT_DIR))

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
CLASSIFICATION_RESULTS = ARTIFACTS_DIR / "classification.json"
DETECTION_RESULTS = ARTIFACTS_DIR / "detection.json"


def prepare_data():
    from src.detection import prepare_detection_dataset

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    counts = prepare_detection_dataset()
    print("Detection data prepared:", counts)


def check_data():
    from src.classification import TRAIN_DIR, VAL_DIR, validate_image_dataset
    from src.detection import count_files, validate_labels

    train_images, train_errors = validate_image_dataset(TRAIN_DIR)
    val_images, val_errors = validate_image_dataset(VAL_DIR)

    print("Classification train images:", len(train_images))
    print("Classification validation images:", len(val_images))

    if train_errors:
        raise ValueError("Some classification images are invalid.")
    if val_errors:
        print(
            "Ignoring invalid validation images because they are not used "
            "in the prepared detection test set:",
            len(val_errors),
        )

    for split in ["train", "valid", "test"]:
        invalid_labels, class_ids = validate_labels(split)
        counts = count_files(split)
        print(
            f"Detection {split}: {counts['images']} images, "
            f"{counts['labels']} labels, class IDs: {class_ids}"
        )
        if invalid_labels:
            raise ValueError(f"Invalid YOLO labels found in {split}.")
        if counts["images"] != counts["labels"]:
            raise ValueError(f"Images and labels do not match in {split}.")


def train_classification():
    from src.classification import run_classification

    model, history, metrics = run_classification()
    results = {
        "metrics": metrics,
        "model": str(PROJECT_DIR / "models" / "brain_tumor_classification_finetuned.keras"),
    }
    with CLASSIFICATION_RESULTS.open("w") as file:
        json.dump(results, file)
    print("Classification training finished.")


def train_detection():
    from src.detection import run_detection

    results = run_detection()
    with DETECTION_RESULTS.open("w") as file:
        json.dump(results, file)
    print("Detection training finished.")


def save_results_in_mlflow():
    from src.mlflow_tracking import log_values, setup_mlflow

    setup_mlflow(
        experiment_name="brain_tumor_models_docker_v3",
        tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///" + str(PROJECT_DIR / "mlflow.db"),
        )
    )

    with CLASSIFICATION_RESULTS.open() as file:
        classification = json.load(file)
    with DETECTION_RESULTS.open() as file:
        detection = json.load(file)

    metrics = {
        "classification_validation_accuracy": classification["metrics"]["validation"]["accuracy"],
        "classification_validation_precision": classification["metrics"]["validation"]["precision"],
        "classification_validation_recall": classification["metrics"]["validation"]["recall"],
        "classification_validation_f1_score": classification["metrics"]["validation"]["f1_score"],
        "classification_test_accuracy": classification["metrics"]["test"]["accuracy"],
        "classification_test_precision": classification["metrics"]["test"]["precision"],
        "classification_test_recall": classification["metrics"]["test"]["recall"],
        "classification_test_f1_score": classification["metrics"]["test"]["f1_score"],
        "classification_fine_tuned_test_accuracy": classification["metrics"]["fine_tuned_test"]["accuracy"],
        "classification_fine_tuned_test_f1_score": classification["metrics"]["fine_tuned_test"]["f1_score"],
        "detection_validation_precision": detection["validation"]["precision"],
        "detection_validation_recall": detection["validation"]["recall"],
        "detection_validation_map50": detection["validation"]["map50"],
        "detection_validation_map50_95": detection["validation"]["map50_95"],
        "detection_test_precision": detection["test"]["precision"],
        "detection_test_recall": detection["test"]["recall"],
        "detection_test_map50": detection["test"]["map50"],
        "detection_test_map50_95": detection["test"]["map50_95"],
    }
    parameters = {
        "classification_model": "InceptionV3",
        "detection_model": "YOLOv8",
    }
    artifacts = [
        classification["model"],
        detection["model_path"],
        str(PROJECT_DIR / "data" / "detection" / "data.yaml"),
        str(CLASSIFICATION_RESULTS),
        str(DETECTION_RESULTS),
    ]

    import mlflow

    with mlflow.start_run(run_name="airflow_pipeline"):
        log_values(parameters, metrics, artifacts)
    print("Results saved in MLflow.")


default_args = {
    "owner": "student",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="brain_tumor_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )
    check = PythonOperator(
        task_id="check_data",
        python_callable=check_data,
    )
    classification = PythonOperator(
        task_id="train_classification",
        python_callable=train_classification,
    )
    detection = PythonOperator(
        task_id="train_detection",
        python_callable=train_detection,
    )
    mlflow = PythonOperator(
        task_id="save_results_in_mlflow",
        python_callable=save_results_in_mlflow,
    )

    prepare >> check
    check >> classification
    check >> detection
    [classification, detection] >> mlflow
