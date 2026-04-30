import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# 1. Generar los mismos datos de prueba
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

# 2. Cargar el modelo
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

# 3. Evaluar
predictions = clf.predict(X)
accuracy = accuracy_score(y, predictions)

# 4. Guardar las métricas en un archivo de texto
with open("results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {accuracy:.2f}\n")

# 5. Generar y guardar una gráfica (Matriz de confusión)
disp = ConfusionMatrixDisplay.from_estimator(clf, X, y, normalize="true", cmap="Blues")
plt.savefig("results/plot.png")
print("Evaluación completada. Métricas y gráfica guardadas en /results.")