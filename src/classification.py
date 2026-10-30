import os
import random

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_DIR = "./data/classification"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR = os.path.join(DATA_DIR, "Val")
MODEL_PATH = "./models/brain_tumor_classification.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return gpus


def validate_image_dataset(split_dir):
    valid_images = []
    invalid_images = []

    for class_name in CLASS_NAMES:
        image_dir = os.path.join(split_dir, class_name, "images")
        if not os.path.isdir(image_dir):
            continue

        for file_name in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file_name)
            extension = os.path.splitext(file_name)[1].lower()

            if not os.path.isfile(file_path) or extension not in VALID_EXTENSIONS:
                invalid_images.append(file_path)
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()
                valid_images.append(file_path)
            except Exception:
                invalid_images.append(file_path)

    return valid_images, invalid_images


def get_label_from_path(image_path):
    for class_name in CLASS_NAMES:
        if os.sep + class_name + os.sep in image_path:
            return class_name
    return None


def explore_dataset(split_dir=TRAIN_DIR):
    valid_images, invalid_images = validate_image_dataset(split_dir)
    counts = {class_name: 0 for class_name in CLASS_NAMES}

    for image_path in valid_images:
        class_name = get_label_from_path(image_path)
        if class_name is not None:
            counts[class_name] += 1

    return {
        "valid_images": len(valid_images),
        "invalid_images": len(invalid_images),
        "class_counts": counts,
    }


def split_dataset(image_paths, test_size=0.3, validation_size=0.5):
    labels = [get_label_from_path(path) for path in image_paths]
    train_paths, temporary_paths, train_labels, temporary_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        temporary_paths,
        temporary_labels,
        test_size=validation_size,
        random_state=42,
        stratify=temporary_labels,
    )
    return (
        train_paths,
        validation_paths,
        test_paths,
        train_labels,
        validation_labels,
        test_labels,
    )


def create_augmentation_preview(image_paths):
    image_path = random.choice(image_paths)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    noise = np.random.randint(-15, 16, image.shape)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    flipped = cv2.flip(image, 1)
    return image, blurred, noisy, flipped


def create_dataset(directory, shuffle=True, batch_size=32):
    image_paths, _ = validate_image_dataset(directory)
    labels = [
        CLASS_NAMES.index(get_label_from_path(image_path))
        for image_path in image_paths
    ]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def load_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    if shuffle:
        dataset = dataset.shuffle(len(image_paths), seed=42)
    return dataset.map(load_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_model(num_classes=len(CLASS_NAMES)):
    base_model = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def train_model(
    train_dir=TRAIN_DIR,
    validation_dir=VAL_DIR,
    epochs=25,
    learning_rate=0.001,
    model_path=MODEL_PATH,
):
    train_dataset = create_dataset(train_dir, shuffle=True)
    validation_dataset = create_dataset(validation_dir, shuffle=False)
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    return model, history, train_dataset, validation_dataset


def evaluate_model(model, dataset):
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_probability = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_probability, axis=1)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def fine_tune_model(model, train_dataset, validation_dataset, epochs=5):
    base_model = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    )
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
    )


def run_classification():
    setup_gpu()
    train_valid, train_invalid = validate_image_dataset(TRAIN_DIR)
    val_valid, val_invalid = validate_image_dataset(VAL_DIR)
    metrics = {
        "train_valid_images": len(train_valid),
        "train_invalid_images": len(train_invalid),
        "validation_valid_images": len(val_valid),
        "validation_invalid_images": len(val_invalid),
    }
    model, history, train_dataset, validation_dataset = train_model()
    metrics["validation"] = evaluate_model(model, validation_dataset)
    fine_tune_model(model, train_dataset, validation_dataset)
    model.save("./models/brain_tumor_classification_finetuned.keras")
    metrics["fine_tuned_validation"] = evaluate_model(model, validation_dataset)
    return model, history, metrics


if __name__ == "__main__":
    run_classification()
