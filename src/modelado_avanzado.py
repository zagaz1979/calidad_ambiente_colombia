# modelado_avanzado.py
# ====================================================
# Modelado Predictivo Avanzado (Regresión Lineal Múltiple)
# para predecir PM10 utilizando Año y PM2.5 como predictores.
# Visualización de resultados en Streamlit.
# ====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

def modelar_avanzado():
    st.header("Modelado Predictivo Avanzado de PM10")

    # Cargar datos filtrados de aire
    df_aire = pd.read_csv('./data/aire_filtrado_caribe.csv')

    # Filtrar Córdoba y pivotear contaminantes
    df_filtrado = df_aire[df_aire['departamento'] == 'CÓRDOBA']
    df_pivot = df_filtrado.pivot_table(
        index='anio',
        columns='variable',
        values='promedio',
        aggfunc='mean'
    ).reset_index()

    st.subheader("Dataset de contaminantes pivotado:")
    st.dataframe(df_pivot.head())

    # Seleccionar variables predictoras y variable objetivo
    X = df_pivot[['anio', 'PM2.5']]
    y = df_pivot['PM10']

    # Eliminar filas con NaN
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[['anio', 'PM2.5']]
    y_clean = data['PM10']

    if data.empty:
        st.warning("No hay datos suficientes para modelar.")
        return

    # Entrenamiento del modelo
    modelo = LinearRegression()
    modelo.fit(X_clean, y_clean)
    y_pred = modelo.predict(X_clean)

    # Evaluación
    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
    r2 = r2_score(y_clean, y_pred)

    st.write(f"**Coeficientes:** {modelo.coef_}")
    st.write(f"**Término independiente:** {modelo.intercept_:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    st.markdown("### Interpretación")
    st.info("""
    - **Coeficientes:** indican el efecto de cada variable predictora en PM10.
    - **R² cercano a 1** indica buen ajuste, **cercano a 0 o negativo** indica mal ajuste.
    - Puede explorarse el uso de modelos no lineales o limpieza avanzada si los resultados no son adecuados.
    """)

    # Visualización
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_clean['anio'], y_clean, color='blue', label='Datos reales')
    ax.plot(X_clean['anio'], y_pred, color='red', label='Predicción (Regresión Lineal)')
    ax.set_title("Predicción de PM10 en CÓRDOBA")
    ax.set_xlabel("Año")
    ax.set_ylabel("PM10 Promedio (µg/m³)")
    ax.legend()
    st.pyplot(fig)

# Ejecución directa para pruebas
if __name__ == "__main__":
    modelar_avanzado()




"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st

sns.set_theme(style="whitegrid")

def modelar_avanzado():
    st.subheader("Modelado Predictivo Avanzado (PM10 ~ Año + PM2.5)")

    df_aire = pd.read_csv('./data/aire_filtrado_caribe.csv')
    df_filtrado = df_aire[(df_aire['departamento'] == 'CÓRDOBA')]

    # Filtrar PM10 y PM2.5 disponibles
    pivot = df_filtrado.pivot_table(index=['anio'], columns='variable', values='promedio', aggfunc='mean').reset_index()
    if 'PM10' not in pivot.columns or 'PM2.5' not in pivot.columns:
        st.warning("No hay suficientes datos de PM10 o PM2.5 en Córdoba para realizar el modelado avanzado.")
        return

    X = pivot[['anio', 'PM2.5']].values
    y = pivot['PM10'].values

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    st.write(f"**Coeficientes:** Año = {modelo.coef_[0]:.4f}, PM2.5 = {modelo.coef_[1]:.4f}")
    st.write(f"**Término independiente:** {modelo.intercept_:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    # Gráfico Predicción vs Real
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(y, y_pred, color='teal')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_title('Predicción vs Real (PM10)')
    ax1.set_xlabel('Valores Reales (PM10)')
    ax1.set_ylabel('Valores Predichos (PM10)')
    st.pyplot(fig1)

    # Gráfico de Residuos
    residuals = y - y_pred
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(y_pred, residuals, color='purple')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_title('Gráfico de Residuos')
    ax2.set_xlabel('Valores Predichos (PM10)')
    ax2.set_ylabel('Residuos')
    st.pyplot(fig2)

    st.success("Modelado avanzado finalizado.")

if __name__ == "__main__":
    modelar_avanzado()
"""
