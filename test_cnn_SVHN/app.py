import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Charger le modèle entraîné
model = tf.keras.models.load_model('mon_modele_svhn.h5')

# Fonction de prétraitement de l'image
def preprocess_image(image, target_size=(32, 32)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    return image

st.title("Application de test du modèle SVHN")

# Chemin du dossier contenant les images de test
test_folder = 'test_photo'

# Vérifier si le dossier existe et contient des images
if os.path.exists(test_folder) and len(os.listdir(test_folder)) > 0:
    for img_file in os.listdir(test_folder):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(test_folder, img_file)
            image = Image.open(image_path)

            st.image(image, caption=f'Image testée : {img_file}', use_column_width=True)

            # Prétraiter et prédire
            processed_image = preprocess_image(image)
            processed_image = np.expand_dims(processed_image, axis=0)  # Ajouter une dimension pour le batch
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)

            st.write(f"Classe prédite pour {img_file} : {predicted_class[0]}")
else:
    st.write("Aucune image trouvée dans le dossier 'test_photo'. Veuillez ajouter des images et rafraîchir.")
