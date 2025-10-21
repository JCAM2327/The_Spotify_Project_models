"""
preprocesamiento.py
-------------------
Preparación de datos LIMPIOS para Machine Learning.
El Data Analyst ya limpió los datos, solo falta transformarlos para sklearn.

IMPORTANTE: Este archivo NO limpia datos (eso ya está hecho).
           Solo transforma para que los modelos puedan usarlos.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def validar_datos_limpios(df):
    """
    Verifica que los datos estén limpios (trabajo del Data Analyst).
    Solo un chequeo rápido, NO limpia nada.
    """
    print("\n🔍 Verificando calidad de datos...")
    
    # Contar nulos
    nulos_totales = df.isnull().sum().sum()
    if nulos_totales > 0:
        print(f"   ⚠️ {nulos_totales} valores nulos encontrados")
        print("   💡 Se rellenarán con media/moda según corresponda")
    else:
        print("   ✅ Sin valores nulos (excelente trabajo del Data Analyst)")
    
    # Verificar outliers marcados
    if 'is_outlier' in df.columns:
        outliers = df['is_outlier'].sum()
        print(f"   📊 {outliers} outliers detectados (marcados por Data Analyst)")
    
    # Verificar calidad
    if 'data_quality_score' in df.columns:
        calidad_promedio = df['data_quality_score'].mean()
        print(f"   ⭐ Calidad promedio: {calidad_promedio:.1f}/100")
    
    return True


def rellenar_nulos_minimos(df):
    """
    Rellena valores nulos SI existen (deberían ser muy pocos).
    """
    nulos = df.isnull().sum()
    
    if nulos.sum() == 0:
        return df
    
    print("\n🔧 Rellenando valores nulos mínimos...")
    
    # Numéricos → media
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"   • {col}: rellenado con media")
    
    # Categóricos → moda
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   • {col}: rellenado con moda")
    
    return df


def codificar_categoricas(df, columnas_categoricas=None):
    """
    Convierte texto a números (género, etc.).
    ESTO SÍ es necesario para Machine Learning.
    """
    if columnas_categoricas is None:
        columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    
    # Filtrar columnas que no queremos codificar (IDs, nombres)
    columnas_excluir = ['track_id', 'track_name', 'artist_name', 'data_source']
    columnas_categoricas = [col for col in columnas_categoricas 
                           if col in df.columns and col not in columnas_excluir]
    
    if not columnas_categoricas:
        print("   ℹ️ No hay columnas categóricas para codificar")
        return df
    
    print(f"\n🔤 Codificando variables categóricas...")
    print(f"   Columnas: {columnas_categoricas}")
    
    for col in columnas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        n_categorias = len(le.classes_)
        print(f"   ✅ {col}: {n_categorias} categorías → 0-{n_categorias-1}")
    
    return df


def escalar_numericas(df, columnas_numericas=None, excluir=None):
    """
    Normaliza rangos de variables numéricas.
    ESTO SÍ es necesario para que el modelo no se confunda.
    """
    if columnas_numericas is None:
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir columnas que no deben escalarse
    if excluir is None:
        excluir = ['track_id', 'release_year']  # IDs y años mejor sin escalar
    
    columnas_numericas = [col for col in columnas_numericas 
                         if col in df.columns and col not in excluir]
    
    if not columnas_numericas:
        print("   ℹ️ No hay columnas numéricas para escalar")
        return df
    
    print(f"\n📏 Escalando variables numéricas...")
    print(f"   Método: StandardScaler (media=0, std=1)")
    print(f"   Columnas ({len(columnas_numericas)}): {columnas_numericas[:5]}...")
    
    scaler = StandardScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
    
    print(f"   ✅ Escalado completado")
    
    return df


def seleccionar_features(df, objetivo="energy", features_base=None):
    """
    Selecciona SOLO las columnas necesarias para el modelo.
    Elimina IDs, nombres, metadata innecesaria.
    """
    print("\n🎯 Seleccionando features para el modelo...")
    
    # Features por defecto (audio features)
    if features_base is None:
        features_base = [
            # Audio features principales
            'danceability', 'loudness', 'valence', 'tempo',
            
            # Variables creadas por Data Analyst (MUY importantes)
            'intensity_weighted',
            
            # Metadata temporal
            'release_year'
        ]
    
    # Verificar cuáles existen
    features_disponibles = [f for f in features_base if f in df.columns]
    features_faltantes = [f for f in features_base if f not in df.columns]
    
    if features_faltantes:
        print(f"   ⚠️ Features no disponibles: {features_faltantes}")
        print(f"   💡 Continuando con features disponibles")
    
    # Agregar género si existe
    if 'main_genre' in df.columns:
        features_disponibles.append('main_genre')
        print(f"   ➕ Agregada: main_genre")
    elif 'genre' in df.columns:
        features_disponibles.append('genre')
        print(f"   ➕ Agregada: genre")
    
    # Verificar objetivo
    if objetivo not in df.columns:
        print(f"   ❌ ERROR: Objetivo '{objetivo}' no encontrado")
        print(f"   Columnas disponibles: {list(df.columns)}")
        return pd.DataFrame()
    
    # Crear dataset con features + objetivo
    columnas_finales = features_disponibles + [objetivo]
    df_modelo = df[columnas_finales].copy()
    
    print(f"\n   ✅ Features seleccionadas ({len(features_disponibles)}):")
    for feat in features_disponibles:
        print(f"      • {feat}")
    print(f"   🎯 Objetivo: {objetivo}")
    print(f"\n   📊 Dataset: {df_modelo.shape[0]:,} filas × {df_modelo.shape[1]} columnas")
    
    return df_modelo


def preparar_pipeline_completo(df, objetivo="energy", features_base=None):
    """
    Pipeline SIMPLIFICADO de preprocesamiento.
    
    ASUME: Datos ya limpios por Data Analyst
    HACE: Solo transformaciones necesarias para ML
    """
    print("="*70)
    print("⚙️ PIPELINE DE PREPROCESAMIENTO PARA MACHINE LEARNING")
    print("="*70)
    print("\n💡 Datos ya limpios por Data Analyst")
    print("   Solo aplicando transformaciones para sklearn/xgboost...")
    
    # 1. Verificar calidad (informativo)
    validar_datos_limpios(df)
    
    # 2. Seleccionar features relevantes
    df_modelo = seleccionar_features(df, objetivo, features_base)
    
    if df_modelo.empty:
        return pd.DataFrame()
    
    # 3. Rellenar nulos mínimos (si existen)
    df_modelo = rellenar_nulos_minimos(df_modelo)
    
    # 4. Codificar categóricas (NECESARIO para ML)
    categoricas = df_modelo.select_dtypes(include=['object']).columns.tolist()
    if categoricas:
        df_modelo = codificar_categoricas(df_modelo, categoricas)
    
    # 5. Escalar numéricas (NECESARIO para ML, excepto objetivo)
    numericas = df_modelo.select_dtypes(include=[np.number]).columns.tolist()
    if objetivo in numericas:
        numericas.remove(objetivo)  # No escalar el objetivo
    
    if numericas:
        df_modelo = escalar_numericas(df_modelo, numericas, excluir=[objetivo])
    
    print("\n" + "="*70)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"   Dataset listo para modelado: {df_modelo.shape[0]:,} filas × {df_modelo.shape[1]} columnas")
    print(f"   Features: {df_modelo.shape[1] - 1}")
    print(f"   Objetivo: {objetivo}")
    
    return df_modelo