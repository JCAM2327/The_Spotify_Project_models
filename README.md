# 🎵 The Spotify Project - ML Specialist

## 🎯 Objetivos del Proyecto

### Preguntas de Negocio Principales

- ¿Qué variables musicales y contextuales explican mejor la intensidad musical de una canción?
- ¿Cómo varía la intensidad según género, década o artista?
- ¿Qué tan precisos pueden ser los modelos de predicción de intensidad?
- ¿Qué patrones emergen al analizar la importancia de variables en los modelos?
> ✅ Todas estas preguntas se abordan en los resultados generados por `main.py` y `reports/resultados_modelos.json`.  
> La importancia de variables se visualiza en `reports/figures/importancia_variables.png`.


### Preguntas Complementarias de Análisis

- ¿Qué características musicales tienen mayor impacto en la predicción de la intensidad?
- ¿Existen diferencias significativas en la intensidad musical entre géneros o décadas?
- ¿Se pueden identificar outliers o patrones inusuales en la intensidad musical?
- ¿Cómo varía la precisión del modelo al usar diferentes combinaciones de variables?
- ¿Qué canciones presentan mayor error de predicción y por qué?
> ✅ Estas preguntas pueden explorarse en los notebooks de Jupyter (`notebooks/03_modelado.ipynb`) y en `reports/resultados_modelos.md`.

### Variables del Proyecto

**Variable Objetivo:**
- `energy` (nivel de intensidad/energía de la canción, 0-1)

**Variables de Entrada Sugeridas (Features):**
- `main_genre` o `genre`
- `tempo`
- `loudness`
- `danceability`
- `valence`
- `acousticness`
- `release_year` o `year`
- Columnas de calidad: `is_complete`, `is_valid_date`, `is_outlier`, `data_quality_score`

> ⚠️ Nota: Se eliminó `intensity_weighted` del conjunto de features para evitar **data leakage (fuga de datos)**.

---

## 📁 Estructura del Proyecto

```
ML_Specialist_Spotify/
│
├── data/
│   ├── raw/                    # Datos originales (no procesados)
│   └── processed/              # Datos limpios listos para modelado
│       └── spotify_clean.csv   # Dataset entregado por Data Analyst
│
├── src/
    ├── __init__.py  
    ├── main.py                 # 🚀 Script principal de automatización
│   ├── carga_dataset.py        # Carga y validación de datos
│   ├── preprocesamiento.py     # Limpieza, codificación y escalado
│   ├── modelado.py             # Entrenamiento y evaluación de modelos
│   ├── interpretacion.py       # Visualización e interpretación
│   └── utils.py                # Funciones auxiliares (guardar resultados, etc.)
│
├── models/
│ ├── regression_lineal.pkl
│ ├── random_forest.pkl
│ └── xgboost.pkl
│
├── notebooks/                  # Notebooks de Jupyter para análisis exploratorio
│   ├── 01_exploracion.ipynb    # Análisis exploratorio inicial (EDA)
│   ├── 02_preprocesamiento.ipynb # Limpieza y preparación de variables
│   └── 03_modelado.ipynb       # Comparación de modelos de ML
│
├── reports/                           # Resultados y reportes generados
│   ├── figures/                
│   │ └── importancia_variables.png    # Gráficos de evaluación (generado automáticamente)
│   └── resultados_modelos.json        # Registro de conclusiones y hallazgos
│
├── requirements.txt            # Dependencias del proyecto
├── app.py                      # Aplicación Streamlit
├── README.md                   # Este archivo
└── .gitignore                  # Exclusiones del control de versiones
```

## 🤖 Modelos de Machine Learning Implementados

Como la variable objetivo `energy` es **numérica continua**, este es un **problema de regresión supervisada**.

### 1. Regresión Lineal (Baseline) 📐

**Descripción:** Modelo estadístico que establece una relación lineal entre variables.

**Ventajas:**
- ✅ Simple, rápida, interpretable
- ✅ Fácil de entender los coeficientes
- ✅ Excelente punto de referencia (baseline)

**Desventajas:**
- ❌ No captura relaciones no lineales
- ❌ Sensible a outliers

**Cuándo usarla:** Como primer modelo para establecer una referencia inicial de rendimiento.

---

### 2. Random Forest Regressor 🌳

**Descripción:** Ensemble de múltiples árboles de decisión que combinan sus predicciones.

**Ventajas:**
- ✅ Maneja no linealidad automáticamente
- ✅ Robusto a outliers
- ✅ Proporciona importancia de variables
- ✅ No requiere normalización de datos

**Desventajas:**
- ❌ Menos interpretable que regresión lineal
- ❌ Requiere más recursos computacionales

**Cuándo usarlo:** Primer modelo robusto para obtener métricas comparables y entender qué variables importan.

---

### 3. XGBoost (Extreme Gradient Boosting) 🚀

**Descripción:** Algoritmo de boosting optimizado que construye árboles secuencialmente.

**Ventajas:**
- ✅ Máxima precisión en la mayoría de casos
- ✅ Maneja automáticamente valores faltantes
- ✅ Regularización incorporada (evita overfitting)
- ✅ Ampliamente usado en competencias de ML

**Desventajas:**
- ❌ Requiere ajuste de hiperparámetros
- ❌ Más tiempo de entrenamiento
- ❌ Menos interpretable

**Cuándo usarlo:** Modelo final o principal cuando necesitas máxima precisión.

---


## 🛠️ Instalación y Configuración

### Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes)
- 4GB RAM mínimo (8GB recomendado)

### Librerías Esenciales

```txt
# Manipulación de datos
pandas >= 2.0.0
numpy >= 1.24.0

# Visualización
matplotlib >= 3.7.0
seaborn >= 0.12.0

# Machine Learning
scikit-learn >= 1.3.0
xgboost >= 1.7.0

# Opcional
lightgbm >= 4.0.0
```

### Pasos de Instalación

#### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/spotify-ml-project.git
cd spotify-ml-project
```

#### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Verificar estructura de datos

Asegúrate de que el archivo `data/processed/spotify_clean.csv` exista y contenga las columnas esperadas:
- `track_name`, `artist`, `genre`, `year`, `tempo`, `danceability`, `energy`, `acousticness`, `loudness`

---

## 🚀 Uso del Proyecto

# Pipeline Automático

Ejecuta todo el proceso con un solo comando:

```bash
# Pipeline completo (entrena los 3 modelos)
python main.py

**El pipeline ejecuta automáticamente:**
1. ✅ Carga y valida los datos desde `data/processed/`
2. ✅ Preprocesa (maneja nulos, codifica géneros)
3. ✅ Entrena 3 modelos (Regresión Lineal, Random Forest, XGBoost)
4. ✅ Evalúa con métricas (MAE, RMSE, R²)
5. ✅ (Opcional) Realiza validación cruzada de 5 folds
6. ✅ Genera gráfico de importancia de variables
7. ✅ Guarda resultados en `reports/resultados_modelos.json`
8. ✅ Muestra resumen en terminal con el mejor modelo

---

### Convenciones de Nombres

- **Scripts:** `snake_case.py` (ej: `carga_dataset.py`)
- **Funciones:** `snake_case()` (ej: `entrenar_modelos()`)
- **Clases:** `PascalCase` (ej: `ModeloPredictor`)
- **Constantes:** `UPPER_CASE` (ej: `RANDOM_STATE = 42`)

## 📚 Recursos Adicionales

### Documentación Oficial

- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Matplotlib](https://matplotlib.org/stable/index.html)


## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

