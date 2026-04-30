import gradio as gr
import skops.io as sio
import pandas as pd

# 1. Leer los tipos de objetos "desconocidos" que tiene el archivo
untrusted_types = sio.get_untrusted_types(file="Model/drug_pipeline.skops")

# 2. Cargar el modelo aprobando explícitamente esa lista de objetos
pipe = sio.load("Model/drug_pipeline.skops", trusted=untrusted_types)

def predict_drug(age, sex, bp, cholesterol, na_to_k):
    # Crear un DataFrame con el formato exacto que espera el pipeline
    data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "BP": [bp],
        "Cholesterol": [cholesterol],
        "Na_to_K": [na_to_k]
    })
    
    # Hacer la predicción
    prediction = pipe.predict(data)[0]
    return prediction

# Definir la interfaz visual basada en la captura de la guía
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, label="Na_to_K")
]

outputs = gr.Textbox(label="output")

app = gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    title="Drug Classification",
    description="Enter the details to correctly identify Drug type?"
)

if __name__ == "__main__":
    app.launch()