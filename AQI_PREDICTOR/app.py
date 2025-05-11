import streamlit as st
import pandas as pd
import joblib   
import numpy as np
from PIL import Image

# --- CONFIGURACIONES DE PÁGINA ---
st.set_page_config(page_title="AQI Predictor", layout="centered")

# --- ESTILO GLOBAL ---
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stSlider > div {
            padding-bottom: 10px;
        }
        h1, h2 {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO PERSONALIZADO CENTRADO ---
st.markdown("<h1 style='text-align: center;'>AQI Predictor</h1>", unsafe_allow_html=True)

# --- INTRODUCCIÓN JUSTIFICADA ---
st.markdown("""
<p style='text-align: justify;'>
Esta aplicación predice la calidad del aire en una región en función de diferentes variables ambientales y socioeconómicas. 
Está diseñada para ayudar a visualizar el nivel de contaminación y tomar medidas preventivas en zonas con alto riesgo ambiental.
</p>
""", unsafe_allow_html=True)

# --- IMAGEN ---
image = Image.open("verde.png")
st.image(image, use_container_width=True)

st.markdown("---")

# --- INPUTS DEL USUARIO ---
st.markdown("### Ingrese los valores ambientales y demográficos:")

temperature = st.slider("🌡️ Temperature (°C)", 13.0, 50.0, 25.0)
no2 = st.slider("🧪 NO2 (ppb)", 5.0, 50.0, 25.0)
so2 = st.slider("🧪 SO2 (ppb)", 0.0, 25.0, 10.0)
co = st.slider("🚗 CO (ppm)", 0.0, 3.0, 1.0)
proximity = st.slider("🏭 Proximidad a zonas industriales (km)", 1.0, 25.0, 10.0)
pop_density = st.slider("👥 Densidad poblacional (personas/km²)", 188.0, 1000.0, 500.0)

# --- CREAR DATAFRAME ---
data = pd.DataFrame({
    'Temperature': [temperature],
    'NO2': [no2],
    'SO2': [so2],
    'CO': [co],
    'Proximity_to_Industrial_Areas': [proximity],
    'Population_Density': [pop_density]
})

# --- PREDICCIÓN ---
if st.button("🔍 Predecir calidad del aire"):
    # Cargar el modelo reentrenado
    model = joblib.load("best_rf_model.pkl")
    pred = int(model.predict(data)[0])

    # Mapear los resultados a etiquetas interpretables
    color_map = {
        0: "#B8D576",
        1: "#FFD95F",
        2: "#EF9651",
        3: "#ff8c8c"
    }

    icon_map = {
        0: "✅",
        1: "⚠️",
        2: "😷",
        3: "☠️"
    }

    description_map = {
        0: "La calidad del aire es satisfactoria y la contaminación del aire representa poco o ningún riesgo para la salud.",
        1: "La calidad del aire es aceptable; sin embargo, puede haber un riesgo moderado para personas extremadamente sensibles.",
        2: "Los miembros de grupos sensibles pueden experimentar efectos en la salud. Es poco probable que el público en general se vea afectado.",
        3: "Condiciones peligrosas. Toda la población puede experimentar efectos graves para la salud. Evite salir al aire libre."
    }

    # Mostrar el resultado
    color = color_map.get(pred, "#ffffff")
    icon = icon_map.get(pred, "❓")
    description = description_map.get(pred, "Sin descripción disponible.")

    st.markdown("---")

    st.markdown(f"""
    <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2>Calidad del Aire: {description} {icon}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar los datos de entrada
    st.markdown("### Datos de entrada:")
    st.dataframe(data)

# --- AUTORES AL FINAL ---
st.markdown("---")
st.markdown("""
<p style='font-size: 14px; text-align: center;'>
<strong>Realizado por:</strong> Leydy Macareo Fuentes y Miguel Angel Vargaz
</p>
""", unsafe_allow_html=True)

# --- PIE DE PÁGINA ---
st.markdown("<p style='text-align: center; font-size: 13px;'>© Unab2025</p>", unsafe_allow_html=True)
