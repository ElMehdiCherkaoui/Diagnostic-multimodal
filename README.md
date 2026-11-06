# Diagnostic multimodal (brain tumor classification + detection)

## Workflow

Install the dependencies from `requirements.txt`, then run the scripts from the
repository root:

```bash
python src/classification.py
python src/detection.py
python src/mlflow_tracking.py
mlflow ui
```

The classification script validates the real `class/images` folders, trains the
InceptionV3-based classifier, evaluates it, fine-tunes it, and saves the models
in `models/`. The detection script prepares the YOLO folders, uses the original
validation data as the independent test set, trains YOLOv8, evaluates both
validation and test data, and saves the best weights in `data/detection/`.

The reusable MLflow functions are in
[`src/mlflow_tracking.py`](src/mlflow_tracking.py). They can log the returned
classification and detection metrics and model artifacts after each run.
Use `run_classification_experiment()` or `run_detection_experiment()` from that
file to train and record a complete experiment.

## Docker services

Install Docker Desktop and enable WSL integration, then run from the project
folder:

```bash
docker compose up -d --build
```

Open the services at:

- Airflow: `http://localhost:8081`
- MLflow: `http://localhost:5001`
- API: `http://localhost:8001/docs`
- Streamlit: `http://localhost:8502`

Stop the services with:

```bash
docker compose down
```