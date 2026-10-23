# ======================================================================
# AUTOMATISATION DU PIPELINE AVEC APACHE AIRFLOW
# ======================================================================
# TIP: Airflow runs best in Docker on Windows. Do this part in Sprint 2.
#
# WHAT TO DO IN THIS FILE:
#
# 1. Créer un DAG Airflow représentant le pipeline de traitement des données et d'entraînement.
# 2. Organiser les tâches dans cet ordre : (a) vérification et préparation des datasets, (b) validation de la structure et de la qualité des données, (c) entraînement du modèle de classification, (d) entraînement du modèle de détection, (e) évaluation des modèles et enregistrement dans MLflow, (f) sauvegarde des modèles et des artefacts.
# 3. Configurer la gestion des erreurs (retries), les journaux d'exécution (logs) et le suivi des statuts des tâches.


# --- Write your code below ---
