#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
------
Aplicación Streamlit profesional para análisis y predicción de energía musical
Basada en los resultados del proyecto ML Specialist - Spotify Music Intensity

Autor: José Mondragón (ML Specialist)
Versión: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ===============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ===============================================================

st.set_page_config(
    page_title="🎵 Predicción de Energía Musical (XGBoost)",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================================
# CONFIGURACIÓN GLOBAL
# ===============================================================

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "xgboost.pkl"

FEATURES = [
    'danceability', 'loudness', 'valence', 'tempo',
    'release_year', 'intensity_weighted', 'main_genre'
]

GENEROS_MAPEO = {
    'Rock': 0, 'Pop': 1, 'Electronic': 2, 'Hip-Hop': 3,
    'Jazz': 4, 'Classical': 5, 'Country': 6, 'R&B': 7,
    'Latin': 8, 'Other': 9
}
GENEROS_INVERSO = {v: k for k, v in GENEROS_MAPEO.items()}

# ===============================================================
# MÉTRICAS REALES DEL MODELO (actualizadas)
# ===============================================================

METRICAS_XGB = {
    "MAE": 0.1046,
    "RMSE": 0.1384,
    "R²": 0.7358,
    "R²_CV_mean": 0.7146,
    "R²_CV_std": 0.0087,
    "descripcion": (
        "XGBoost fue el modelo con mejor desempeño global, "
        "explicando el 73.6% de la varianza en la intensidad musical. "
        "Demostró estabilidad con baja desviación estándar (±0.0087) "
        "en validación cruzada."
    )
}

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado desde disco (con caché)."""
    if not MODEL_FILE.exists():
        st.error("❌ No se encontró el modelo entrenado.")
        st.info("Ejecuta `python main.py` para entrenar y guardar el modelo XGBoost.")
        return None
    return joblib.load(MODEL_FILE)


def preparar_datos(df):
    """Prepara los datos para predicción."""
    df_prep = df.copy()
    if 'main_genre' in df_prep.columns and df_prep['main_genre'].dtype == 'object':
        df_prep['main_genre'] = df_prep['main_genre'].map(GENEROS_MAPEO)
        df_prep['main_genre'].fillna(GENEROS_MAPEO['Other'], inplace=True)
    for col in FEATURES:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
    return df_prep


def interpretar_energia(valor):
    """Devuelve etiqueta y color según el valor de energía."""
    if valor >= 0.8:
        return "🔥 Muy Alta", "#FF4B4B"
    elif valor >= 0.6:
        return "⚡ Alta", "#FFA500"
    elif valor >= 0.4:
        return "🎵 Media", "#FFD700"
    elif valor >= 0.2:
        return "🎹 Baja", "#90EE90"
    else:
        return "🌙 Muy Baja", "#87CEEB"

# ===============================================================
# CARGA DEL MODELO
# ===============================================================

modelo = cargar_modelo()
st.title("🎵 Predictor de Energía Musical - XGBoost")
st.markdown("### 🤖 Proyecto ML Specialist | Spotify Music Intensity")
st.markdown("---")

# ===============================================================
# INFORMACIÓN DEL MODELO
# ===============================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{METRICAS_XGB['R²']:.3f}")
col2.metric("MAE", f"{METRICAS_XGB['MAE']:.3f}")
col3.metric("RMSE", f"{METRICAS_XGB['RMSE']:.3f}")
col4.metric("CV Std", f"±{METRICAS_XGB['R²_CV_std']:.4f}")

st.info(METRICAS_XGB["descripcion"])

# ===============================================================
# ENTRADA DE DATOS
# ===============================================================

st.header("1️⃣ Ingreso de Datos")

tab1, tab2 = st.tabs(["📁 Subir CSV", "✍️ Entrada Manual"])

with tab1:
    archivo = st.file_uploader("📂 Sube tu archivo CSV con canciones", type=['csv'])
    if archivo:
        df = pd.read_csv(archivo)
        st.success(f"✅ Archivo cargado: {len(df)} canciones")
        st.dataframe(df.head(5), use_container_width=True)
    else:
        df = None

with tab2:
    st.write("### ✍️ Ingresar una canción manualmente")
    c1, c2 = st.columns(2)
    with c1:
        track = st.text_input("🎵 Nombre", "Mi Canción Nueva")
        artist = st.text_input("🎤 Artista", "Artista Ejemplo")
        dance = st.slider("💃 Danceability", 0.0, 1.0, 0.75, 0.01)
        loud = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0, 0.1)
    with c2:
        val = st.slider("😊 Valence", 0.0, 1.0, 0.68, 0.01)
        tempo = st.number_input("🥁 Tempo (BPM)", 40, 250, 128)
        year = st.number_input("📅 Año", 1960, 2025, 2024)
        inten = st.slider("⚡ Intensidad ponderada", 0.0, 1.0, 0.72, 0.01)
    genre = st.selectbox("🎸 Género musical", list(GENEROS_MAPEO.keys()))

    if st.button("➕ Agregar canción manual"):
        df = pd.DataFrame([{
            'track_name': track,
            'artist_name': artist,
            'danceability': dance,
            'loudness': loud,
            'valence': val,
            'tempo': tempo,
            'release_year': year,
            'intensity_weighted': inten,
            'main_genre': genre
        }])
        st.success("✅ Canción agregada correctamente")
        st.dataframe(df)

# ===============================================================
# PREDICCIÓN
# ===============================================================

st.markdown("---")
st.header("2️⃣ Predicción de Energía")

if df is not None and modelo is not None:
    if st.button("🚀 Generar Predicción", use_container_width=True):
        try:
            df_prep = preparar_datos(df)

            # === FIX: asegurar coincidencia de columnas ===
            columnas_esperadas = getattr(modelo, "feature_names_in_", FEATURES)
            X = df_prep[[col for col in columnas_esperadas if col in df_prep.columns]].copy()

            if list(X.columns) != list(columnas_esperadas):
                for col in columnas_esperadas:
                    if col not in X.columns:
                        X[col] = 0
                X = X[columnas_esperadas]

            # Predicción
            pred = modelo.predict(X)
            df['energy_predicted'] = pred
            df['nivel_energia'], df['color'] = zip(*[interpretar_energia(p) for p in pred])

            st.success("✅ Predicciones generadas correctamente")
            # Mostrar resultados con manejo de columnas opcionales
            columnas_a_mostrar = [col for col in ['track_name', 'artist_name', 'energy_predicted', 'nivel_energia'] if col in df.columns]
            st.dataframe(df[columnas_a_mostrar], use_container_width=True)

            # =========================================================
            # VISUALIZACIONES DE RESULTADOS
            # =========================================================
            st.markdown("---")
            st.header("3️⃣ Visualizaciones Analíticas")

            fig_hist = px.histogram(df, x='energy_predicted', nbins=30,
                                    title="Distribución de Energía Predicha",
                                    color_discrete_sequence=['#1DB954'])
            st.plotly_chart(fig_hist, use_container_width=True)

            if 'main_genre' in df.columns:
                df_gen = df.copy()
                if df_gen['main_genre'].dtype != 'object':
                    df_gen['main_genre'] = df_gen['main_genre'].map(GENEROS_INVERSO)
                fig_gen = px.bar(
                    df_gen.groupby('main_genre')['energy_predicted'].mean().reset_index(),
                    x='main_genre', y='energy_predicted',
                    color='energy_predicted', color_continuous_scale='Viridis',
                    title="Promedio de Energía por Género"
                )
                st.plotly_chart(fig_gen, use_container_width=True)

            if 'release_year' in df.columns:
                fig_line = px.line(
                    df.groupby('release_year')['energy_predicted'].mean().reset_index(),
                    x='release_year', y='energy_predicted',
                    title="Tendencia Temporal de Energía Musical",
                    markers=True
                )
                fig_line.update_traces(line_color='#1DB954', line_width=3)
                st.plotly_chart(fig_line, use_container_width=True)

            # =========================================================
            # INTERPRETACIÓN AUTOMÁTICA
            # =========================================================
            st.markdown("---")
            st.header("4️⃣ Interpretación Automática")
            promedio = df['energy_predicted'].mean()
            etiqueta, color = interpretar_energia(promedio)
            st.markdown(f"**Nivel promedio de energía:** {etiqueta} ({promedio:.3f})")

            if promedio > 0.7:
                st.success("🎧 Tu dataset contiene canciones mayormente intensas y enérgicas.")
            elif promedio < 0.4:
                st.info("🌙 Tu dataset tiene tendencia hacia canciones más suaves y relajadas.")
            else:
                st.warning("🎵 Tu dataset es balanceado en intensidad musical.")

        except Exception as e:
            st.error(f"❌ Error al generar la predicción: {e}")
else:
    st.info("👆 Sube un archivo CSV o ingresa manualmente una canción para comenzar.")

# ===============================================================
# FOOTER
# ===============================================================

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
    <p>🎧 Proyecto ML Specialist - Spotify Intensity Analysis</p>
</div>
""", unsafe_allow_html=True)
