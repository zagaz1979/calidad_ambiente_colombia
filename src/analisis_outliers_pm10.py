import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")

def analisis_outliers_pm10():
    st.subheader("Análisis de Outliers y Regresión en PM10 - CÓRDOBA")

    df_aire = pd.read_csv('./data/aire_filtrado_caribe.csv')
    contaminante_objetivo = 'PM10'
    df_filtrado = df_aire[
        (df_aire['departamento'] == 'CÓRDOBA') &
        (df_aire['variable'] == contaminante_objetivo)
    ].copy()

    if df_filtrado.empty:
        st.warning("No hay datos de PM10 en CÓRDOBA.")
        return

    # Modelo original
    X = df_filtrado[['anio']].values
    y = df_filtrado['promedio'].values

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    st.write("### Modelo Original (con todos los datos)")
    st.write(f"Coeficiente: `{modelo.coef_[0]:.4f}`")
    st.write(f"Término independiente: `{modelo.intercept_:.4f}`")
    st.write(f"RMSE: `{rmse:.4f}`")
    st.write(f"R²: `{r2:.4f}`")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(X, y, color='blue', label='Datos reales')
    ax1.plot(X, y_pred, color='red', label='Regresión lineal')
    ax1.set_title("Modelo Original: PM10 en CÓRDOBA")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("PM10 Promedio (µg/m³)")
    ax1.legend()
    st.pyplot(fig1)

    # Outliers con IQR
    Q1 = df_filtrado['promedio'].quantile(0.25)
    Q3 = df_filtrado['promedio'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    outliers = df_filtrado[
        (df_filtrado['promedio'] < limite_inf) |
        (df_filtrado['promedio'] > limite_sup)
    ]

    st.write(f"Se detectaron `{outliers.shape[0]}` outliers con el método IQR.")
    st.dataframe(outliers[['anio', 'promedio']])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.boxplot(x=df_filtrado['promedio'], ax=ax2)
    ax2.set_title("Boxplot de PM10 en CÓRDOBA (con Outliers)")
    ax2.set_xlabel("PM10 Promedio (µg/m³)")
    st.pyplot(fig2)

    # Modelo sin outliers
    df_sin_outliers = df_filtrado[
        (df_filtrado['promedio'] >= limite_inf) &
        (df_filtrado['promedio'] <= limite_sup)
    ]
    X_sin = df_sin_outliers[['anio']].values
    y_sin = df_sin_outliers['promedio'].values

    modelo_sin = LinearRegression()
    modelo_sin.fit(X_sin, y_sin)
    y_pred_sin = modelo_sin.predict(X_sin)

    rmse_sin = np.sqrt(mean_squared_error(y_sin, y_pred_sin))
    r2_sin = r2_score(y_sin, y_pred_sin)

    st.write("### Modelo Sin Outliers")
    st.write(f"Coeficiente: `{modelo_sin.coef_[0]:.4f}`")
    st.write(f"Término independiente: `{modelo_sin.intercept_:.4f}`")
    st.write(f"RMSE: `{rmse_sin:.4f}`")
    st.write(f"R²: `{r2_sin:.4f}`")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(X_sin, y_sin, color='green', label='Datos sin outliers')
    ax3.plot(X_sin, y_pred_sin, color='red', label='Regresión lineal (sin outliers)')
    ax3.set_title("Modelo de PM10 en CÓRDOBA (Sin Outliers)")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("PM10 Promedio (µg/m³)")
    ax3.legend()
    st.pyplot(fig3)

    st.success("Análisis de outliers y modelado completados correctamente.")