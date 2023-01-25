import pandas as pd
import random

# Générer 100 lignes de données aléatoires
data = []
for i in range(100):
  year = random.randint(2015, 2020)
  population = random.randint(1000, 1500)
  fertility_rate = random.uniform(0.5, 0.75)
  newborns = population * fertility_rate
  data.append([year, population, fertility_rate, newborns])

# Créer un DataFrame Pandas à partir de vos données
df = pd.DataFrame(data, columns=['year', 'population', 'fertility_rate', 'newborns'])

# Enregistrer le DataFrame dans un fichier CSV
df.to_csv('africa_newborns.csv', index=False)
