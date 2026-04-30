import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Cargar datos de prueba (usaremos un dataset dummy muy simple)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

# 2. Entrenar el modelo
print("Entrenando el modelo...")
clf = RandomForestClassifier(max_depth=2, random_state=42)
clf.fit(X, y)

# 3. Guardar el modelo
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("¡Modelo guardado exitosamente como model.pkl!")