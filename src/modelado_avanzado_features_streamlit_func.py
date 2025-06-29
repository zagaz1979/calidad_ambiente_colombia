"""
modelado_avanzado_features_streamlit_func.py

Función lista para integrarse en app.py para modelado predictivo con Streamlit,
usando regresión lineal, polinómica y Random Forest,
permitiendo selección de features basadas en correlación,
y generando métricas, predicciones 2024-2025 y exportación de gráficos.

Autor: 
Fecha: 2025-06-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

def modelado_avanzado_features_streamlit():
    st.title("Modelado Predictivo Avanzado con Features")

    st.info("Cargando dataset './data/aire_filtrado_caribe.csv' para análisis y modelado avanzado.")
    df = pd.read_csv('./data/aire_filtrado_caribe.csv')

    st.subheader("Visualización Inicial del Dataset Filtrado")
    st.dataframe(df.head())

    st.subheader("Selección de Variables Predictoras")
    opciones = ['anio', 'dias_excedencias', 'percentil_98', 'maximo', 'minimo', 'mediana', 'suma', 'n_datos']
    features_seleccionadas = st.multiselect(
        "Selecciona las variables que usarás para el modelado predictivo:",
        opciones,
        default=['anio', 'dias_excedencias', 'percentil_98', 'maximo', 'minimo']
    )

    if len(features_seleccionadas) == 0:
        st.warning("Selecciona al menos una variable predictora para continuar.")
        return

    X = df[features_seleccionadas]
    y = df['promedio']

    # Imputación rápida por mediana si hay NaN
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Regresión Lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    y_pred_lineal = modelo_lineal.predict(X)
    r2_lineal = r2_score(y, y_pred_lineal)
    rmse_lineal = np.sqrt(mean_squared_error(y, y_pred_lineal))

    # Regresión Polinómica (grado 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    y_pred_poly = modelo_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

    # Random Forest
    modelo_rf = RandomForestRegressor(random_state=42)
    modelo_rf.fit(X, y)
    y_pred_rf = modelo_rf.predict(X)
    r2_rf = r2_score(y, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))

    st.subheader("Resultados de los Modelos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lineal R²", f"{r2_lineal:.4f}")
    col1.metric("Lineal RMSE", f"{rmse_lineal:.4f}")
    col2.metric("Polinómica R²", f"{r2_poly:.4f}")
    col2.metric("Polinómica RMSE", f"{rmse_poly:.4f}")
    col3.metric("Random Forest R²", f"{r2_rf:.4f}")
    col3.metric("Random Forest RMSE", f"{rmse_rf:.4f}")

    # Predicciones 2024-2025
    st.subheader("Predicciones para 2024 y 2025")
    futuro = pd.DataFrame({'anio': [2024, 2025]})
    for col in features_seleccionadas:
        if col != 'anio':
            futuro[col] = [X[col].median(), X[col].median()]

    futuro_lineal = modelo_lineal.predict(futuro[features_seleccionadas])
    futuro_poly = modelo_poly.predict(poly.transform(futuro[features_seleccionadas]))
    futuro_rf = modelo_rf.predict(futuro[features_seleccionadas])

    pred_df = pd.DataFrame({
        'Año': [2024, 2025],
        'Lineal': futuro_lineal,
        'Polinómica': futuro_poly,
        'Random Forest': futuro_rf
    })
    st.dataframe(pred_df)

    # Gráficos y exportación
    Path('./outputs').mkdir(exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.scatter(y, y_pred_lineal, label='Lineal')
    plt.scatter(y, y_pred_poly, label='Polinómica')
    plt.scatter(y, y_pred_rf, label='Random Forest')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('Comparación de Modelos Predictivos')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./outputs/comparacion_modelos_features.png')
    st.success("Gráfico comparativo guardado automáticamente en ./outputs/comparacion_modelos_features.png")
    st.image('./outputs/comparacion_modelos_features.png', caption="Comparación de Modelos Predictivos", use_column_width=True)

    st.info("Proceso de modelado predictivo avanzado con selección de features completado.")
