import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


app = FastAPI(title="Brain Tumor API")

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASSIFICATION_MODEL = None
DETECTION_MODEL = None


def get_classification_model():
    global CLASSIFICATION_MODEL

    if CLASSIFICATION_MODEL is None:
        model_path = PROJECT_DIR / "models" / "brain_tumor_classification_finetuned.keras"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Classification model not found: {model_path}",
            )
        try:
            import tensorflow as tf

            CLASSIFICATION_MODEL = tf.keras.models.load_model(model_path)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=503,
                detail="Classification model could not be loaded.",
            ) from error

    return CLASSIFICATION_MODEL


def get_detection_model():
    global DETECTION_MODEL

    if DETECTION_MODEL is None:
        model_path = PROJECT_DIR / "data" / "detection" / "brain_tumor_yolo_best.pt"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Detection model not found: {model_path}",
            )
        try:
            from ultralytics import YOLO

            DETECTION_MODEL = YOLO(model_path)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise HTTPException(
                status_code=503,
                detail=f"Detection model could not be loaded: {error}",
            ) from error

    return DETECTION_MODEL


async def read_image(file):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        return image
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Invalid image file.") from error


@app.get("/")
def home():
    return {"message": "Brain Tumor API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image = await read_image(file)
    model = get_classification_model()
    image = np.array(image.resize((224, 224)), dtype=np.float32)
    predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    best_class = int(np.argmax(predictions))

    return {
        "class": CLASS_NAMES[best_class],
        "confidence": float(predictions[best_class]),
        "probabilities": {
            name: float(predictions[index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = await read_image(file)
    model = get_detection_model()
    try:
        result = model.predict(image, device="cpu", verbose=False)[0]
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail="Detection failed while running the YOLO model.",
        ) from error
    detections = []

    for box, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        class_id = int(class_id)
        detections.append(
            {
                "class": result.names[class_id],
                "class_id": class_id,
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
        )

    return {"detections": detections}
