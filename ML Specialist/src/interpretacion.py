"""
interpretacion.py
-----------------
Funciones para interpretar resultados de los modelos entrenados y visualizar
la importancia de variables.

Script para visualización e interpretación de resultados de modelos
Aquí puedes graficar importancia de variables, errores y comparar predicciones.
Útil para entender el comportamiento de los modelos y comunicar hallazgos.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def importancia_variables(modelo, X):
    """Genera un gráfico de importancia de variables (si el modelo lo permite)."""
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importancias)), importancias[indices])
        plt.xticks(range(len(importancias)), X.columns[indices], rotation=90)
        plt.title("Importancia de Variables")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Este modelo no tiene atributo 'feature_importances_'.")
