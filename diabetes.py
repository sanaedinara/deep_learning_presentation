from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
dataset = loadtxt('candidates-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
# séparation des variables d'entrées (X) et les variables de sorties (Y)
# de l'indice 0 à l'indice 7 de notre fichier dataset
X = dataset[:,0:8]
y = dataset[:,8]
# définir l'architecture de notre modèle avec Keras
# en utilisant le modèle Sequential qui  est une pile linéaire de couches
model = Sequential()

# On utilise la méthode .add() pour spécifier
# les couches que vous voulez que votre modèle ait.
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile le modèle Keras
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuster le modèle keras sur le jeu de données
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# Faire des prédictions de classe avec le modèle
predictions = (model.predict(X) > 0.5).astype(int)

# résumer les 5 premiers cas,
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


 #l'age,nombre de fois enceinte(femme),indice de masse corporelle(poids,taiie),insulene serique de 2h
 #epaisseur du pli cutané du triceps,pression artérielle diastoliqu;fonction généalogique du diabéte,
 #concentration de glucos a 2h