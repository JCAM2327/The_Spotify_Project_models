"""
modelado.py
-----------
Funciones para entrenar y evaluar modelos de ML de forma robusta.
Incluye cross-validation y prevención de data leakage.
"""

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def dividir_datos(df, objetivo="energy", test_size=0.2, random_state=42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    # Evitar leakage eliminando columnas que contengan información directa del objetivo
    X = df.drop(columns=[objetivo, "intensity_weighted"], errors="ignore")
    y = df[objetivo]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def entrenar_modelos(X_train, y_train):
    """Entrena varios modelos y devuelve diccionario."""
    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    for nombre, modelo in modelos.items():
        print(f"⚙️ Entrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

    return modelos

def evaluar_modelos(modelos, X_test, y_test, cv=5):
    """
    Evalúa modelos entrenados usando test set y cross-validation.
    Devuelve diccionario con métricas promedio y std.
    """
    resultados = {}
    for nombre, modelo in modelos.items():
        print(f"\n📊 Evaluando: {nombre}")

        # Predicciones en test set
        pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        # Cross-validation para R²
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(modelo, X_test, y_test, cv=kf, scoring='r2')
        r2_cv_mean = np.mean(cv_scores)
        r2_cv_std = np.std(cv_scores)

        resultados[nombre] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "R²_CV_mean": r2_cv_mean,
            "R²_CV_std": r2_cv_std
        }

        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        print(f"R² CV ({cv}-fold): {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")

    return resultados
