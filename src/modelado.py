"""
modelado.py

Modelado predictivo simple sobre Calidad del Aire y Agua en Colombia.
- Predicción de contaminantes (ejemplo: PM10) con regresión lineal.
- Predicción de IRCA en el tiempo con regresión lineal.

Autor: [César García, Luis Rodriguez y Rosalinda Parra]
Fecha: 2025-06-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def modelar_calidad_aire(streamlit_mode=False):
    if streamlit_mode and not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado en este entorno.")

    df_aire = pd.read_csv('./data/aire_filtrado_caribe.csv')
    contaminante_objetivo = 'PM10'
    departamento_objetivo = 'CÓRDOBA'

    df_filtrado = df_aire[
        (df_aire['departamento'] == departamento_objetivo) &
        (df_aire['variable'] == contaminante_objetivo)
    ]

    if df_filtrado.empty:
        mensaje = f"No hay datos de {contaminante_objetivo} en {departamento_objetivo}."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    X = df_filtrado[['anio']].values
    y = df_filtrado['promedio'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    if streamlit_mode:
        st.subheader(f"Predicción: {contaminante_objetivo} en {departamento_objetivo}")
        st.write(f"**Coeficiente:** {modelo.coef_[0]:.4f}")
        st.write(f"**Término independiente:** {modelo.intercept_:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        y_full_pred = modelo.predict(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción de {contaminante_objetivo} en {departamento_objetivo}")
        ax.set_xlabel("Año")
        ax.set_ylabel(f"{contaminante_objetivo} Promedio (µg/m³)")
        ax.legend()
        st.pyplot(fig)

    else:
        print(f"Coeficiente: {modelo.coef_[0]:.4f}")
        print(f"Término independiente: {modelo.intercept_:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        y_full_pred = modelo.predict(X)
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label='Datos reales')
        plt.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        plt.title(f"Predicción de {contaminante_objetivo} en {departamento_objetivo}")
        plt.xlabel("Año")
        plt.ylabel(f"{contaminante_objetivo} Promedio (µg/m³)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def modelar_calidad_agua(streamlit_mode=False):
    if streamlit_mode and not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit no está instalado en este entorno.")

    df_agua = pd.read_csv('./data/agua_filtrada_caribe.csv')
    departamento_objetivo = 'CÓRDOBA'

    df_filtrado = df_agua[df_agua['departamento'] == departamento_objetivo]
    df_grouped = df_filtrado.groupby('anio')['irca'].mean().reset_index()

    X = df_grouped[['anio']].values
    y = df_grouped['irca'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    if streamlit_mode:
        st.subheader(f"Predicción: IRCA en {departamento_objetivo}")
        st.write(f"**Coeficiente:** {modelo.coef_[0]:.4f}")
        st.write(f"**Término independiente:** {modelo.intercept_:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        y_full_pred = modelo.predict(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='green', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción del IRCA Promedio en {departamento_objetivo}")
        ax.set_xlabel("Año")
        ax.set_ylabel("IRCA Promedio (%)")
        ax.legend()
        st.pyplot(fig)

    else:
        print(f"Coeficiente: {modelo.coef_[0]:.4f}")
        print(f"Término independiente: {modelo.intercept_:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

        y_full_pred = modelo.predict(X)
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='green', label='Datos reales')
        plt.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        plt.title(f"Predicción del IRCA Promedio en {departamento_objetivo}")
        plt.xlabel("Año")
        plt.ylabel("IRCA Promedio (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    modelar_calidad_aire()
    modelar_calidad_agua()