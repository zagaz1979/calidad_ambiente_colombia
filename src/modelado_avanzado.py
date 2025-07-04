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