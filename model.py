import os
import pandas as pd
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# 1. Cargar y preparar datos
drug_df = pd.read_csv("data/drug.csv")
drug_df = drug_df.sample(frac=1)

X = drug_df.drop("Drug", axis=1)
y = drug_df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 2. Configurar el Pipeline
cat_col = ["Sex", "BP", "Cholesterol"]
num_col = ["Age", "Na_to_K"]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
    ]
)

# 3. Entrenar y guardar
print("Entrenando el pipeline...")
pipe.fit(X_train, y_train)

# ¡ESTA ES LA LÍNEA MÁGICA QUE CREARÁ LA CARPETA!
os.makedirs("Model", exist_ok=True)

# Guardar usando skops
sio.dump(pipe, "Model/drug_pipeline.skops")
print("¡Modelo guardado exitosamente en Model/drug_pipeline.skops!")