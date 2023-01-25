import tensorflow as tf
import cv2

# Charger le modèle pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Charger l'image
image = cv2.imread('image1.jpg')
image = cv2.resize(image, (224, 224))


# Prédire l'étiquette de l'image
predictions = model.predict(image)

# Trouver l'étiquette la plus probable
label = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

# Afficher l'étiquette prédite
print(label)