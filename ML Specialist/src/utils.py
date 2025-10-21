"""
utils.py
--------
Funciones auxiliares: guardar resultados, exportar métricas y logs.

Funciones auxiliares para el proyecto ML Specialist
Aquí puedes agregar utilidades como guardado/carga de modelos, métricas personalizadas, etc.
"""

import json
from pathlib import Path

def guardar_resultados(resultados, ruta="reports/resultados_modelos.json"):
    """Guarda las métricas de evaluación en un archivo JSON."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(resultados, f, indent=4)
    print(f"💾 Resultados guardados en {ruta}")
