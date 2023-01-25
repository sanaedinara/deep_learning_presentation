import tensorflow as neural

# Charger et préparer les données
(x_train, y_train), (x_test, y_test) = neural.keras.datasets.africa_newborns()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Définir le modèle
model = neural.keras.models.Sequential([
  neural.keras.layers.Flatten(input_shape=(28, 28)),
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

# Évaluer le modèle
model.evaluate(x_test, y_test)
