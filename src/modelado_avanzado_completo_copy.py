import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

# ==============================
# Modelado avanzado completo
# ==============================
def modelar_avanzado_completo():
    st.title("Modelado Avanzado de Calidad del Aire")
    st.write("Comparación de Regresión Lineal, Regresión Polinómica y Random Forest para predicción de PM10 en CÓRDOBA.")

    # Cargar datos filtrados
    df = pd.read_csv('./data/aire_filtrado_caribe.csv')

    # Filtros para Córdoba y PM10
    df_filtrado = df[(df['departamento'] == 'CÓRDOBA') & (df['variable'] == 'PM10')]

    if df_filtrado.empty:
        st.warning("No hay datos de PM10 para Córdoba disponibles.")
        return

    X = df_filtrado[['anio']].values
    y = df_filtrado['promedio'].values

    # =========================
    # Regresión Lineal
    # =========================
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    y_pred_lineal = modelo_lineal.predict(X)

    # =========================
    # Regresión Polinómica (grado 2)
    # =========================
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    y_pred_poly = modelo_poly.predict(X_poly)

    # =========================
    # Random Forest
    # =========================
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)
    y_pred_rf = modelo_rf.predict(X)

    # =========================
    # Métricas
    # =========================
    rmse_lineal = np.sqrt(mean_squared_error(y, y_pred_lineal))
    r2_lineal = r2_score(y, y_pred_lineal)

    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
    r2_poly = r2_score(y, y_pred_poly)

    rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
    r2_rf = r2_score(y, y_pred_rf)

    # =========================
    # Predicciones 2024-2025
    # =========================
    X_futuro = np.array([[2024], [2025]])
    X_futuro_poly = poly.transform(X_futuro)
    pred_lineal_futuro = modelo_lineal.predict(X_futuro)
    pred_poly_futuro = modelo_poly.predict(X_futuro_poly)
    pred_rf_futuro = modelo_rf.predict(X_futuro)

    # =========================
    # Visualización Comparativa
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Datos reales')
    ax.plot(X, y_pred_lineal, color='red', linestyle='--', label='Lineal')
    ax.plot(X, y_pred_poly, color='green', linestyle='-.', label='Polinómica (grado 2)')
    ax.plot(X, y_pred_rf, color='purple', linestyle='-', label='Random Forest')
    ax.set_title("Comparación de Modelos de Predicción de PM10 en CÓRDOBA")
    ax.set_xlabel("Año")
    ax.set_ylabel("PM10 Promedio (µg/m³)")
    ax.legend()
    st.pyplot(fig)
    fig.savefig("./outputs/comparacion_modelos_pm10_cordoba.png")

    # =========================
    # Resultados en Streamlit
    # =========================
    st.subheader("Resultados de los Modelos")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lineal R²", f"{r2_lineal:.4f}")
        st.metric("Lineal RMSE", f"{rmse_lineal:.4f}")
    with col2:
        st.metric("Polinómica R²", f"{r2_poly:.4f}")
        st.metric("Polinómica RMSE", f"{rmse_poly:.4f}")
    with col3:
        st.metric("Random Forest R²", f"{r2_rf:.4f}")
        st.metric("Random Forest RMSE", f"{rmse_rf:.4f}")

    st.subheader("Predicciones para 2024 y 2025")
    pred_df = pd.DataFrame({
        "Año": [2024, 2025],
        "Lineal": pred_lineal_futuro,
        "Polinómica": pred_poly_futuro,
        "Random Forest": pred_rf_futuro
    })
    st.table(pred_df)

    st.info("Los gráficos se han guardado automáticamente en la carpeta ./outputs para tu informe final.")

# Ejecución directa opcional
if __name__ == "__main__":
    modelar_avanzado_completo()
