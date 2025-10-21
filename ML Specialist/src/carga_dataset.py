#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carga_dataset.py
----------------
Script para cargar y validar el dataset procesado proporcionado
por el Data Analyst. Descarga desde Google Drive y organiza en carpetas.
"""

import pandas as pd
import sys
from pathlib import Path

# ===============================================================
# CONFIGURACIÓN
# ===============================================================

# 🔗 ID del archivo de Google Drive
FILE_ID = "1mghdutXbc7woBVSqwUx5S1axGO2chxlu"

# 📁 Estructura de carpetas
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_FILE = PROCESSED_DIR / "spotify_music_intensity_clean.csv"

# Columnas esperadas (ACTUALIZADAS según tu dataset real)
COLUMNAS_ESPERADAS = {
    "identificacion": ["track_id", "track_name", "artist_name"],
    "audio_features": ["energy", "loudness", "danceability", "valence", "tempo"],
    "metadata": ["release_year", "release_decade", "genre", "main_genre"],
    "intensidad": ["intensity_weighted", "intensity_simple", "intensity_complex"],
    "calidad": ["is_complete", "is_valid_date", "data_quality_score"]
}

# ===============================================================
# FUNCIONES
# ===============================================================

def crear_estructura_carpetas():
    """Crea la estructura de carpetas del proyecto"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Estructura de carpetas creada:")
    print(f"   {DATA_DIR}/")
    print(f"   └── {PROCESSED_DIR}/")


def cargar_datos_gdown(file_id, output_file):
    """
    Carga el dataset desde Google Drive usando gdown.
    """
    try:
        import gdown
        print("📥 Descargando datos desde Google Drive con gdown...")
        
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Descargar en la ubicación correcta
        gdown.download(url, str(output_file), quiet=False)
        
        print(f"✅ Archivo descargado: {output_file}")
        df = pd.read_csv(output_file)
        print(f"✅ Datos cargados correctamente.")
        print(f"   Filas: {df.shape[0]:,}, Columnas: {df.shape[1]}")
        return df
        
    except ImportError:
        print("❌ La biblioteca 'gdown' no está instalada.")
        print("📦 Instálala ejecutando: pip install gdown")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        print("\n💡 SOLUCIONES:")
        print("1. Verifica que el archivo sea público (Cualquiera con el enlace)")
        print("2. Verifica el ID del archivo en Google Drive")
        print("3. Intenta descargar manualmente y colocarlo en data/processed/")
        return pd.DataFrame()


def cargar_datos_local(archivo):
    """
    Carga el dataset desde un archivo local.
    """
    try:
        print(f"📂 Cargando datos desde archivo local: {archivo}")
        df = pd.read_csv(archivo)
        print(f"✅ Datos cargados correctamente.")
        print(f"   Filas: {df.shape[0]:,}, Columnas: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {archivo}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return pd.DataFrame()


def validar_columnas(df, columnas_esperadas):
    """Verifica que el dataset contenga las columnas necesarias."""
    if df.empty:
        print("⚠️ El DataFrame está vacío. Revisa el origen de los datos.")
        return False

    print("\n🔍 Validando estructura del dataset...")
    print(f"   Columnas encontradas: {list(df.columns)}")
    
    # Contar cuántas columnas esperadas están presentes
    total_esperadas = sum(len(cols) for cols in columnas_esperadas.values())
    encontradas = []
    
    for categoria, cols in columnas_esperadas.items():
        cols_presentes = [col for col in cols if col in df.columns]
        if cols_presentes:
            encontradas.extend(cols_presentes)
    
    print(f"\n✅ Columnas presentes: {len(encontradas)}/{total_esperadas}")
    
    # Mostrar columnas por categoría
    for categoria, cols in columnas_esperadas.items():
        cols_presentes = [col for col in cols if col in df.columns]
        cols_faltantes = [col for col in cols if col not in df.columns]
        
        if cols_presentes:
            print(f"   {categoria.upper()}: {len(cols_presentes)}/{len(cols)} presentes")
        if cols_faltantes:
            print(f"      ⚠️ Faltantes: {cols_faltantes}")
    
    return True


def mostrar_info(df):
    """Muestra información básica del dataset."""
    if df.empty:
        return
    
    print("\n" + "="*70)
    print("📊 INFORMACIÓN DEL DATASET")
    print("="*70)
    print(f"\n📏 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    print("\n📋 Primeras 5 filas:")
    print(df.head())
    
    print("\n🔢 Tipos de datos:")
    print(df.dtypes)
    
    print("\n❓ Valores nulos por columna:")
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(nulos[nulos > 0])
    else:
        print("   ✅ No hay valores nulos")
    
    print("\n📈 Estadísticas de columnas numéricas:")
    print(df.describe())


# ===============================================================
# EJECUCIÓN PRINCIPAL
# ===============================================================

if __name__ == "__main__":
    print("="*70)
    print("🎵 CARGADOR DE DATASET MUSICAL")
    print("="*70 + "\n")
    
    # Crear estructura de carpetas
    crear_estructura_carpetas()
    
    # Intentar cargar desde archivo local primero
    if OUTPUT_FILE.exists():
        print(f"\n💾 Archivo local encontrado: {OUTPUT_FILE}")
        respuesta = input("¿Usar archivo local existente? (s/n): ")
        if respuesta.lower() == 's':
            df = cargar_datos_local(OUTPUT_FILE)
        else:
            df = cargar_datos_gdown(FILE_ID, OUTPUT_FILE)
    else:
        # Si no existe, descargar desde Drive
        df = cargar_datos_gdown(FILE_ID, OUTPUT_FILE)
    
    # Validar y mostrar información
    if not df.empty:
        if validar_columnas(df, COLUMNAS_ESPERADAS):
            mostrar_info(df)
            print(f"\n💾 Dataset guardado en: {OUTPUT_FILE}")
        else:
            print("\n⚠️ El dataset tiene diferencias con la estructura esperada.")
            print("   Pero se cargó correctamente y está listo para usar.")
    else:
        print("\n❌ No se pudo cargar el dataset.")
        print("\n📝 PASOS PARA SOLUCIONAR:")
        print("1. Instala gdown: pip install gdown")
        print("2. Verifica que el archivo sea público en Google Drive")
        print(f"3. O descarga manualmente y colócalo en: {OUTPUT_FILE}")