import os, time, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# ======================================================================
# CLASSIFICATION DES TUMEURS CEREBRALES AVEC GOOGLENET
# ======================================================================
# TIP: Test your code in notebook.ipynb first, then move the working code here as functions. ATTENTION: Keras n'a pas InceptionV1 dans keras.applications (seulement InceptionV3). Cherche des poids GoogLeNet préentraînés (port Keras, ou conversion depuis torchvision).
#
# WHAT TO DO IN THIS FILE:
#
# 1. Charger les images, vérifier extensions et intégrité (try-except), écarter les fichiers invalides.
# 2. Explorer les classes (noms de dossiers), countplot, afficher un échantillon par classe.
# 3. Répartir les images en trois ensembles : Train (70 %), Validation (15 %) et Test (15 %), puis compter et afficher le nombre d'images dans chaque ensemble.
# 4. Appliquer des techniques d'augmentation des données (blur, noise, flip) uniquement sur les données d'entraînement, afin d'enrichir les échantillons et de réduire les déséquilibres entre les classes.
# 5. Charger avec image_dataset_from_directory() (redimensionnement, batch, shuffle=True pour le train), puis normaliser selon le modèle.
# 6. Charger le modèle préentraîné GoogLeNet (InceptionV1), geler ses couches de base, remplacer sa tête de classification par une nouvelle adaptée au nombre de classes, puis entraîner uniquement cette dernière (Feature Extraction).
# 7. Déterminer les hyperparamètres d'entraînement : learning rate, fonction de perte (loss) et optimiseur (optimizer).
# 8. Entraîner le modèle sur l'ensemble d'entraînement et suivre ses performances sur l'ensemble de validation.
# 9. Évaluer le modèle : accuracy, précision (precision), rappel (recall) et F1-score.
# 10. Analyser les résultats avec une matrice de confusion et comparer les performances sur les ensembles d'entraînement et de test.
# 11. Dégeler les dernières couches du modèle préentraîné, poursuivre le fine-tuning avec un faible learning rate, réévaluer le modèle et comparer avec le modèle du Feature Extraction.
# 12. Sauvegarder le modèle entraîné.


# --- Write your code below ---

def validate_image_dataset(split_dir, class_names, valid_extensions):
    valid_images = []
    invalid_images = []

    for class_name in class_names:
        folder_path = os.path.join(split_dir, class_name, "images")

        for file_name in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file_name)

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(file_name)[1].lower()

            if extension not in valid_extensions:
                invalid_images.append(file_path)
                continue

            try:
                with Image.open(file_path) as img:
                    img.load()

                valid_images.append(file_path)

            except Exception:
                invalid_images.append(file_path)

    return valid_images, invalid_images

def get_label_from_path(image_path,class_names):
    for class_name in class_names:
        if class_name in image_path:
            return class_name
    return None

def split_dataset(X,class_names):
    X_train, X_temp = train_test_split(
        X, test_size=0.3, random_state=42
    )

    X_val, X_test = train_test_split(
        X_temp, test_size=0.5, random_state=42
    )
    
    y_train = [get_label_from_path(path, class_names) for path in X_train]
    y_val = [get_label_from_path(path, class_names) for path in X_val]
    y_test = [get_label_from_path(path, class_names) for path in X_test]

    return X_train, X_val, X_test, y_train, y_val, y_test

def augmentation(image):
    image_path = random.choice(image)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.randint(-15, 16, img.shape)
    noise_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    flipped_img = cv2.flip(img, 1)
    return blurred_img, noise_img, flipped_img

def create_dataset_and_normalize(directory, shuffle=True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(224, 224),
        batch_size=32,
        shuffle=shuffle
    )
    
    normalization = tf.keras.layers.Rescaling(1./255)

    dataset = dataset.map(
        lambda images, labels: (normalization(images), labels)
    )

    return dataset

def create_model():
    base_model = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        4,
        activation="softmax"
    )(x)
    model = tf.keras.Model(inputs, outputs)
    return model