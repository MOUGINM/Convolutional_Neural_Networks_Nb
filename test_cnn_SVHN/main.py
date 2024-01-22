import numpy as np
from scipy.io import loadmat
import tensorflow as tf

def load_svhn(path):
    data = loadmat(path)
    x = np.array(data['X'])
    y = data['y'].flatten()
    y[y == 10] = 0  # Remplacer 10 par 0
    x = np.transpose(x, (3, 0, 1, 2))  # Réorganiser les dimensions
    return x, y

# Chemins vers les fichiers de données
train_path = 'train_32x32.mat'
test_path = 'test_32x32.mat'

# Charger les données
x_train, y_train = load_svhn(train_path)
x_test, y_test = load_svhn(test_path)

# Normalisation des images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Conversion des labels en catégories pour l'utilisation avec 'sparse_categorical_crossentropy'
# Pas besoin de conversion si 'sparse_categorical_crossentropy' est utilisé

# Création du modèle CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Perte: {loss}, Précision: {accuracy}')

# Enregistrement du modèle
model.save('mon_modele_svhn.h5')

# Charger le modèle
model_charge = tf.keras.models.load_model('mon_modele_svhn.h5')

# Utiliser le modèle chargé pour faire des prédictions ou évaluer
loss, accuracy = model_charge.evaluate(x_test, y_test)
print(f'Perte rechargée: {loss}, Précision rechargée: {accuracy}')
