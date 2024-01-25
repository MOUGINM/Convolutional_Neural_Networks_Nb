# Projet de Classification des Chiffres SVHN avec un CNN

Ce projet implémente un modèle Convolutional Neural Network (CNN) pour la classification des chiffres du dataset Street View House Numbers (SVHN). Il utilise TensorFlow et Streamlit pour créer une interface utilisateur interactive permettant de tester le modèle avec des images personnalisées.

## Contenu du Répertoire

- `main.py`: Le script principal qui charge, entraîne, évalue et enregistre le modèle CNN.
- `app.py`: Le script Streamlit pour l'interface utilisateur permettant de télécharger des images et de faire des prédictions.
- `make_nb.py`: Un script utilitaire pour extraire et sauvegarder des images SVHN à utiliser comme données de test.
- `mon_modele_svhn.h5`: Le modèle CNN entraîné, enregistré au format HDF5.

# SVHN Digit Classification with CNN

## Project Overview
This project develops a Convolutional Neural Network (CNN) model to classify digits from the Street View House Numbers (SVHN) dataset. Leveraging TensorFlow for deep learning and Streamlit for an interactive interface, this project allows users to test the model with custom images.

## Repository Contents

- `main.py`: Main script for loading, training, evaluating, and saving the CNN model.
- `app.py`: Streamlit script for the user interface, allowing image uploads and predictions.
- `make_nb.py`: Utility script for extracting and saving SVHN images for test data.
- `mon_modele_svhn.h5`: Trained CNN model, saved in HDF5 format.

### Prerequisites
- Python 3.x
- TensorFlow
- Streamlit

### Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run Streamlit app: `streamlit run app.py`.
