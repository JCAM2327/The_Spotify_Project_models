#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
-------
Pipeline completo de Machine Learning para predicción de energía musical.
Versión corregida para evitar data leakage y usar cross-validation.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ajustar rutas
if Path.cwd().name == 'src':
    PROJECT_ROOT = Path.cwd().parent
else:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Importar módulos del proyecto
from preprocesamiento import preparar_pipeline_completo
from modelado import dividir_datos, entrenar_modelos, evaluar_modelos
from interpretacion import importancia_variables
from utils import guardar_resultados

# ===============================================================
# CONFIGURACIÓN DE RUTAS
# ===============================================================
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILE = DATA_DIR / "spotify_music_intensity_clean.csv"
RESULTADOS_FILE = REPORTS_DIR / "resultados_modelos.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================
# CONFIGURACIÓN DEL MODELO
# ===============================================================
OBJETIVO = "energy"  # Energía de la canción (0-1)

FEATURES_BASE = [
    'danceability',
    'loudness',
    'valence',
    'tempo',
    'release_year'
]  # Eliminamos intensity_weighted para evitar leakage

USAR_MUESTRA = True
TAMAÑO_MUESTRA = 50000

# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================
def verificar_recursos():
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024**3)
        print(f"🧠 RAM disponible: {ram_gb:.2f} GB")
        if ram_gb < 2:
            print("⚠️ Poca RAM (<2GB). Usar muestra pequeña")
            return False
        return True
    except ImportError:
        print("ℹ️ Instala psutil: pip install psutil")
        return True

def cargar_dataset():
    if not DATASET_FILE.exists():
        print(f"❌ Dataset no encontrado: {DATASET_FILE}")
        sys.exit(1)
    df = pd.read_csv(DATASET_FILE)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df

def muestrear_datos(df, tamaño_muestra):
    if len(df) <= tamaño_muestra:
        return df
    if USAR_MUESTRA:
        df_muestra = df.sample(n=tamaño_muestra, random_state=42)
        print(f"🎲 Usando muestra de {len(df_muestra):,} canciones")
        return df_muestra
    return df

def explorar_datos(df):
    print("\n🔍 Exploración rápida")
    if 'main_genre' in df.columns:
        print("\n🎸 Top 5 géneros:")
        print(df['main_genre'].value_counts().head())
    if 'release_decade' in df.columns:
        print("\n📅 Canciones por década:")
        print(df['release_decade'].value_counts().sort_index())
    if OBJETIVO in df.columns:
        print(f"\n⚡ Distribución de {OBJETIVO}:")
        print(df[OBJETIVO].describe())

def guardar_modelo(modelo, nombre, carpeta=MODELS_DIR):
    try:
        import joblib
        ruta = carpeta / f"{nombre}.pkl"
        joblib.dump(modelo, ruta)
        print(f"💾 Modelo guardado: {ruta}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar modelo: {e}")

# ===============================================================
# PIPELINE PRINCIPAL
# ===============================================================
def main():
    print("="*70)
    print("🎵 PREDICCIÓN DE ENERGÍA MUSICAL CON MACHINE LEARNING")
    print("="*70)

    # PASO 1: Verificar recursos
    verificar_recursos()

    # PASO 2: Cargar y explorar dataset
    df = cargar_dataset()
    explorar_datos(df)
    df = muestrear_datos(df, TAMAÑO_MUESTRA)

    # PASO 3: Preprocesar
    df_modelo = preparar_pipeline_completo(df, OBJETIVO, FEATURES_BASE)
    if df_modelo.empty:
        print("❌ Error en preprocesamiento")
        sys.exit(1)

    # PASO 4: Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(df_modelo, OBJETIVO)
    print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # PASO 5: Entrenar modelos
    modelos = entrenar_modelos(X_train, y_train)
    for nombre, modelo in modelos.items():
        nombre_archivo = nombre.lower().replace(' ', '_').replace('ó','o')
        guardar_modelo(modelo, nombre_archivo)

    # PASO 6: Evaluar modelos con cross-validation
    resultados = evaluar_modelos(modelos, X_test, y_test, cv=5)
    guardar_resultados(resultados, str(RESULTADOS_FILE))

    # Importancia de features
    print("\n📊 Importancia de variables (Random Forest)")
    importancia_variables(modelos["Random Forest"], X_train)
    figura_path = FIGURES_DIR / "importancia_variables.png"
    plt.savefig(figura_path, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfico guardado: {figura_path}")

    # Mejor modelo
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['R²'])
    print(f"\n🏆 Mejor modelo: {mejor_modelo[0]} | R²: {mejor_modelo[1]['R²']:.4f}")

    # Próximos pasos para análisis de negocio
    print("\n💡 Próximos pasos:")
    print("   • Revisar reports/resultados_modelos.json")
    print("   • Analizar importancia de features y tendencias por género/década")
    print("   • Explorar outliers y errores de predicción para preguntas de negocio")

# ===============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
