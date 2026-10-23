# ======================================================================
# API FASTAPI
# ======================================================================
# TIP: Run with: uvicorn api.main:app --reload
#
# WHAT TO DO IN THIS FILE:
#
# 1. Créer l'application FastAPI et charger les deux modèles (classification + détection) au démarrage.
# 2. Endpoint POST /classify : recevoir une image, retourner la classe prédite, le score de confiance et les probabilités.
# 3. Endpoint POST /detect : recevoir une image, retourner les boîtes englobantes, les classes et les scores de confiance.
# 4. Gérer les erreurs : fichier invalide, requête incorrecte, échec d'exécution.
# 5. Documenter les endpoints (Swagger disponible sur /docs).


# --- Write your code below ---
