import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from io import BytesIO
import os

def ejecutar_modelado_avanzado_features():
    st.header("Modelado Predictivo Avanzado con Selección de Features")

    df = pd.read_csv('./data/aire_filtrado_caribe.csv')

    st.subheader("Dataset de Entrada")
    st.dataframe(df.head())

    # Selección de variables numéricas relevantes
    variables_numericas = ['anio', 'promedio', 'suma', 'n_datos', 'mediana', 'percentil_98', 'maximo', 'minimo', 'dias_excedencias']
    df_numerico = df[variables_numericas].dropna()

    # Correlación con PM10 únicamente
    if 'variable' in df.columns:
        df_pm10 = df[df['variable'] == 'PM10']
    else:
        st.error("La columna 'variable' no está en el dataset.")
        return

    corr_matrix = df_pm10[variables_numericas].corr()
    st.subheader("Matriz de Correlación")
    st.dataframe(corr_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Selección de features con correlación > 0.3 en valor absoluto con 'promedio' (target)
    corr_target = corr_matrix['promedio'].abs()
    selected_features = corr_target[corr_target > 0.3].index.tolist()
    selected_features = [f for f in selected_features if f != 'promedio']

    st.subheader("Features Seleccionadas")
    st.write(selected_features)

    if not selected_features:
        st.warning("No se encontraron features con correlación significativa. Se utilizará únicamente el año.")
        selected_features = ['anio']

    X = df_pm10[selected_features].values
    y = df_pm10['promedio'].values

    # ===========================
    # Modelos
    # ===========================
    results = {}

    # Lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    y_pred_lineal = modelo_lineal.predict(X)
    results['Lineal'] = {
        'r2': r2_score(y, y_pred_lineal),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_lineal))
    }

    # Polinómico grado 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    y_pred_poly = modelo_poly.predict(X_poly)
    results['Polinómico'] = {
        'r2': r2_score(y, y_pred_poly),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_poly))
    }

    # Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)
    y_pred_rf = modelo_rf.predict(X)
    results['Random Forest'] = {
        'r2': r2_score(y, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_rf))
    }

    # ===========================
    # Resultados Visuales
    # ===========================
    st.subheader("Resultados de los Modelos")
    for modelo, metrica in results.items():
        st.markdown(f"**{modelo} R²:** {metrica['r2']:.4f}")
        st.markdown(f"**{modelo} RMSE:** {metrica['rmse']:.4f}")

    # ===========================
    # Predicciones 2024-2025
    # ===========================
    st.subheader("Predicciones para 2024 y 2025")
    años_pred = pd.DataFrame({'anio': [2024, 2025]})

    pred_df = pd.DataFrame({'Año': [2024, 2025]})

    if 'anio' in selected_features and len(selected_features) == 1:
        X_pred_lineal = años_pred.values
        X_pred_poly = poly.transform(años_pred.values)
        X_pred_rf = años_pred.values
    else:
        # Usar la media de otras variables y año futuro
        mean_values = df_pm10[selected_features].mean().to_dict()
        pred_rows = []
        for año in [2024, 2025]:
            row = [año if col == 'anio' else mean_values[col] for col in selected_features]
            pred_rows.append(row)
        X_pred_lineal = np.array(pred_rows)
        X_pred_poly = poly.transform(X_pred_lineal)
        X_pred_rf = X_pred_lineal

    pred_df['Lineal'] = modelo_lineal.predict(X_pred_lineal)
    pred_df['Polinómica'] = modelo_poly.predict(X_pred_poly)
    pred_df['Random Forest'] = modelo_rf.predict(X_pred_rf)

    st.dataframe(pred_df)

    # ===========================
    # Exportación de gráficos
    # ===========================
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
    ax_pred.plot(df_pm10['anio'], y, 'bo', label='Datos Reales')
    ax_pred.plot(df_pm10['anio'], y_pred_lineal, 'r-', label='Lineal')
    ax_pred.plot(df_pm10['anio'], y_pred_poly, 'g--', label='Polinómica')
    ax_pred.plot(df_pm10['anio'], y_pred_rf, 'y-.', label='Random Forest')
    ax_pred.set_xlabel('Año')
    ax_pred.set_ylabel('PM10 Promedio (µg/m³)')
    ax_pred.legend()
    ax_pred.set_title('Comparación de Modelos Predictivos')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'comparacion_modelos_pm10.png')
    fig_pred.savefig(output_path)
    st.pyplot(fig_pred)

    st.success(f"Los gráficos se han guardado automáticamente en {output_path} para tu informe final.")

# Permite ejecución directa opcional para pruebas locales
if __name__ == "__main__":
    ejecutar_modelado_avanzado_features()