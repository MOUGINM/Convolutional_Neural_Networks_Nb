# Projet de Classification des Chiffres SVHN avec un CNN

Ce projet implémente un modèle Convolutional Neural Network (CNN) pour la classification des chiffres du dataset Street View House Numbers (SVHN). Il utilise TensorFlow et Streamlit pour créer une interface utilisateur interactive permettant de tester le modèle avec des images personnalisées.

## Contenu du Répertoire

- `main.py`: Le script principal qui charge, entraîne, évalue et enregistre le modèle CNN.
- `app.py`: Le script Streamlit pour l'interface utilisateur permettant de télécharger des images et de faire des prédictions.
- `make_nb.py`: Un script utilitaire pour extraire et sauvegarder des images SVHN à utiliser comme données de test.
- `mon_modele_svhn.h5`: Le modèle CNN entraîné, enregistré au format HDF5.
