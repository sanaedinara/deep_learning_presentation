import pandas as pd
import numpy as np
import tensorflow as neural

from sklearn.model_selection import train_test_split

# Charger les données à partir d'un fichier CSV
data = pd.read_csv('africa_newborns.csv')

# Séparer les données en entrées et cibles
X = data.drop('newborns', axis=1)
y = data['newborns']

# Convertir les données en tableaux NumPy
X = X.values
y = y.values

# Découper les données en ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#(x_train, y_train), (x_test, y_test) = neural.keras.datasets.africa_newborns()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Définir le modèle
model = neural.keras.models.Sequential([
  neural.keras.layers.Flatten(input_shape=(12, 28, 28)),
  neural.keras.layers.Dense(128, activation='relu'),
  neural.keras.layers.Dropout(0.2),
  neural.keras.layers.Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5)

# Normaliser les données
#X_train = X_train / 255.0
#X_test = X_test / 255.0
