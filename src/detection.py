# ======================================================================
# DETECTION DES TUMEURS CEREBRALES AVEC YOLOV8
# ======================================================================
# TIP: Test your code in notebook.ipynb first, then move the working code here as functions.
#
# WHAT TO DO IN THIS FILE:
#
# 1. Afficher un échantillon d'images avec leurs boîtes englobantes (bounding boxes) et les classes correspondantes.
# 2. Vérifier, pour chaque image, l'existence d'un fichier .txt correspondant dans le répertoire des annotations (comparer les noms de base des fichiers).
# 3. Organiser les données dans les dossiers images et labels, en respectant les ensembles train, valid et test.
# 4. Copier chaque image avec son fichier d'annotation vers l'ensemble approprié.
# 5. Écarter les images sans annotation et les fichiers d'annotation sans image associée.
# 6. Vérifier la validité du format des annotations YOLO et la cohérence des classes.
# 7. Compter les images et les fichiers d'annotation dans chaque ensemble.
# 8. Créer un fichier data.yaml (chemins train/val/test, nombre de classes, noms des classes).
# 9. Définir les paramètres d'augmentation des données dans la configuration d'entraînement de YOLOv8.
# 10. Entraîner le modèle YOLOv8 avec des hyperparamètres adaptés au dataset et aux ressources disponibles.
# 11. Évaluer : précision, rappel, mAP@0.5 et mAP@0.5:0.95.
# 12. Analyser les résultats et vérifier la généralisation du modèle sur l'ensemble de test.
# 13. Sauvegarder le modèle entraîné et les artefacts nécessaires à son utilisation.


# --- Write your code below ---
