import os
import pandas as pd
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Cargar datos de prueba
drug_df = pd.read_csv("data/drug.csv")
X = drug_df.drop("Drug", axis=1)
y = drug_df.Drug.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# 2. Cargar el modelo seguro
untrusted_types = sio.get_untrusted_types(file="Model/drug_pipeline.skops")
pipe = sio.load("Model/drug_pipeline.skops", trusted=untrusted_types)

# 3. Evaluar
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# 4. Guardar resultados
os.makedirs("results", exist_ok=True)
with open("results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}\n")

# 5. Generar gráfica
cm = ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=pipe.classes_)
plt.savefig("results/model_results.png", dpi=120)
print("Evaluación completada.")