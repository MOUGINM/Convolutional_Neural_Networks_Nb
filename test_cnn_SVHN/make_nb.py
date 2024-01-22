import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

# Nombre d'images à extraire
nombre_images = 10  # Modifie cette valeur selon tes besoins

# Chemin vers le fichier de données SVHN
svhn_path = 'train_32x32.mat'  # Assure-toi que ce chemin est correct

# Charger le dataset SVHN
data = loadmat(svhn_path)
x = np.array(data['X'])

# Créer le dossier 'test_photo' s'il n'existe pas
dossier_test = 'test_photo'
if not os.path.exists(dossier_test):
    os.makedirs(dossier_test)

# Extraire et sauvegarder les images
for i in range(nombre_images):
    img = x[:,:,:,i]
    img_pil = Image.fromarray(img)
    chemin_image = os.path.join(dossier_test, f'svhn_{i}.png')
    img_pil.save(chemin_image, 'PNG')  # Utiliser un format de haute qualité comme PNG

print(f"{nombre_images} images SVHN de haute qualité ont été sauvegardées dans le dossier '{dossier_test}'.")
