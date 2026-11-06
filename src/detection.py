import os
import random
import shutil

from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DATA_DIR = "./data/classification"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VAL_DIR = os.path.join(DATA_DIR, "Val")
DETECTION_DIR = "./data/detection"
DATA_YAML = os.path.join(DETECTION_DIR, "data.yaml")


def create_detection_folders():
    for split in ["train", "valid", "test"]:
        for folder in ["images", "labels"]:
            os.makedirs(os.path.join(DETECTION_DIR, split, folder), exist_ok=True)


def clear_folder(folder):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def show_image_with_labels(image_path, label_path, class_names=CLASS_NAMES):
    import cv2
    import matplotlib.pyplot as plt

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    plt.imshow(image)

    with open(label_path, "r") as file:
        for line in file:
            class_id, center_x, center_y, box_width, box_height = map(
                float, line.split()
            )
            x = (center_x - box_width / 2) * width
            y = (center_y - box_height / 2) * height
            box = plt.Rectangle(
                (x, y),
                box_width * width,
                box_height * height,
                fill=False,
                color="red",
            )
            plt.gca().add_patch(box)
            plt.text(x, y, class_names[int(class_id)], color="red")

    plt.axis("off")
    plt.show()


def check_labels(split_dir):
    missing_labels = []
    extra_labels = []
    for class_name in CLASS_NAMES:
        image_dir = os.path.join(split_dir, class_name, "images")
        label_dir = os.path.join(split_dir, class_name, "labels")
        if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
            continue
        image_names = {
            os.path.splitext(name)[0] for name in os.listdir(image_dir)
        }
        label_names = {
            os.path.splitext(name)[0] for name in os.listdir(label_dir)
        }
        missing_labels.extend(sorted(image_names - label_names))
        extra_labels.extend(sorted(label_names - image_names))
    return missing_labels, extra_labels


def copy_pair(image_path, label_path, target_folder, class_name, image_name):
    new_name = class_name + "_" + image_name
    shutil.copy2(image_path, os.path.join(target_folder, "images", new_name))
    label_name = os.path.splitext(new_name)[0] + ".txt"
    shutil.copy2(label_path, os.path.join(target_folder, "labels", label_name))


def get_pairs(source_dir, class_name):
    image_dir = os.path.join(source_dir, class_name, "images")
    label_dir = os.path.join(source_dir, class_name, "labels")
    pairs = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(
            label_dir, os.path.splitext(image_name)[0] + ".txt"
        )
        if os.path.exists(label_path):
            pairs.append((image_path, label_path, image_name))
    return pairs


def prepare_detection_dataset(validation_size=0.15, random_state=42):
    create_detection_folders()
    for split in ["train", "valid", "test"]:
        clear_folder(os.path.join(DETECTION_DIR, split, "images"))
        clear_folder(os.path.join(DETECTION_DIR, split, "labels"))

    counts = {"train": 0, "valid": 0, "test": 0}
    random.seed(random_state)
    for class_name in CLASS_NAMES:
        pairs = get_pairs(TRAIN_DIR, class_name)
        random.shuffle(pairs)
        split_index = int(len(pairs) * (1 - validation_size))
        split_pairs = {
            "train": pairs[:split_index],
            "valid": pairs[split_index:],
        }
        for split, selected_pairs in split_pairs.items():
            for image_path, label_path, image_name in selected_pairs:
                copy_pair(
                    image_path,
                    label_path,
                    os.path.join(DETECTION_DIR, split),
                    class_name,
                    image_name,
                )
                counts[split] += 1

        for image_path, label_path, image_name in get_pairs(VAL_DIR, class_name):
            copy_pair(
                image_path,
                label_path,
                os.path.join(DETECTION_DIR, "test"),
                class_name,
                image_name,
            )
            counts["test"] += 1
    return counts


def remove_unmatched_files(split):
    split_dir = os.path.join(DETECTION_DIR, split)
    image_dir = os.path.join(split_dir, "images")
    label_dir = os.path.join(split_dir, "labels")
    image_names = {os.path.splitext(name)[0] for name in os.listdir(image_dir)}
    label_names = {os.path.splitext(name)[0] for name in os.listdir(label_dir)}
    for name in image_names - label_names:
        for image_name in os.listdir(image_dir):
            if os.path.splitext(image_name)[0] == name:
                os.remove(os.path.join(image_dir, image_name))
    for name in label_names - image_names:
        os.remove(os.path.join(label_dir, name + ".txt"))


def validate_labels(split):
    labels_dir = os.path.join(DETECTION_DIR, split, "labels")
    invalid_labels = []
    class_ids = set()
    for file_name in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, file_name), "r") as file:
            for line_number, line in enumerate(file, start=1):
                values = line.split()
                if len(values) != 5:
                    invalid_labels.append((file_name, line_number))
                    continue
                class_id, x, y, width, height = map(float, values)
                class_ids.add(int(class_id))
                valid = (
                    class_id.is_integer()
                    and 0 <= class_id < len(CLASS_NAMES)
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                )
                if not valid:
                    invalid_labels.append((file_name, line_number))
    return invalid_labels, sorted(class_ids)


def count_files(split):
    split_dir = os.path.join(DETECTION_DIR, split)
    images = len(os.listdir(os.path.join(split_dir, "images")))
    labels = len(os.listdir(os.path.join(split_dir, "labels")))
    return {"images": images, "labels": labels}


def create_data_yaml():
    data_yaml = (
        "path: ./data/detection\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 4\n"
        "names: [Glioma, Meningioma, No_Tumor, Pituitary]\n"
    )
    with open(DATA_YAML, "w") as file:
        file.write(data_yaml)


def train_model(
    epochs=5,
    image_size=320,
    batch_size=8,
    device=None,
    model_path="yolov8n.pt",
):
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print("YOLO device:", device)

    model = YOLO(model_path)
    return model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        workers=0,
        device=device,
        project="./runs/detection",
        name="brain_tumor",
        degrees=10,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
    )


def evaluate_model(model, split, device=None):
    import torch

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        device=device,
        workers=0,
    )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def save_model(results):
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    saved_model_path = os.path.join(DETECTION_DIR, "brain_tumor_yolo_best.pt")
    shutil.copyfile(best_model_path, saved_model_path)
    shutil.copyfile(
        DATA_YAML,
        os.path.join(DETECTION_DIR, "brain_tumor_data.yaml"),
    )
    return saved_model_path


def run_detection():
    from ultralytics import YOLO

    counts = prepare_detection_dataset()
    for split in ["train", "valid", "test"]:
        remove_unmatched_files(split)
    create_data_yaml()
    results = train_model()
    model = YOLO(os.path.join(results.save_dir, "weights", "best.pt"))
    validation_metrics = evaluate_model(model, "val")
    test_metrics = evaluate_model(model, "test")
    saved_model_path = save_model(results)
    return {
        "counts": counts,
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": saved_model_path,
    }


if __name__ == "__main__":
    run_detection()
