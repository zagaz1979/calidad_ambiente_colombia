import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st # Importación directa, ya que siempre estará disponible en Streamlit

sns.set_theme(style="whitegrid")

# Diccionario para mapear contaminantes y parámetros atmosféricos a sus unidades de medida.
# Esto se utiliza para generar etiquetas de ejes Y dinámicas y precisas en los gráficos.
UNIDADES_CONTAMINANTES = {
    'PM10': 'µg/m³',    # Partículas con diámetro aerodinámico de 10 micrómetros o menos.
                        # Contaminante primario, asociado a polvo, construcción, tráfico.
    'O3': 'µg/m³',      # Ozono troposférico. Contaminante secundario, formado por reacción
                        # entre NOx y compuestos orgánicos volátiles con luz solar.
    'PST': 'µg/m³',     # Partículas Suspendidas Totales. Partículas de cualquier tamaño
                        # que permanecen suspendidas en el aire.
    'P': 'hPa o mbar',  # Presión atmosférica. Presión ejercida por el aire sobre la superficie.
    'PM2.5': 'µg/m³',   # Partículas con diámetro aerodinámico de 2.5 micrómetros o menos.
                        # Contaminante primario, más peligroso por su pequeño tamaño.
    'TAire2': '°C',     # Temperatura del aire. Medición de la temperatura ambiente.
    'SO2': 'µg/m³',     # Dióxido de azufre. Contaminante asociado a la quema de combustibles
                        # fósiles (carbón, diésel).
    'NO2': 'µg/m³',     # Dióxido de nitrógeno. Contaminante emitido principalmente por
                        # vehículos y procesos industriales.
    'CO': 'ppm o mg/m³',# Monóxido de carbono. Gas tóxico incoloro, producto de la combustión incompleta.
    'HAire2': '%',      # Humedad relativa del aire. Cantidad de vapor de agua en el aire
                        # respecto al máximo posible a esa temperatura.
    'DViento': 'grados',# Dirección del viento. Dirección desde la que sopla el viento.
    'RGlobal': 'W/m²',  # Radiación solar global. Cantidad total de energía solar recibida
                        # sobre una superficie horizontal.
    'VViento': 'm/s o km/h' # Velocidad del viento. Velocidad con la que se mueve el aire.
                            # Puede influir en la dispersión de contaminantes.
}


def modelar_calidad_aire(df_aire, streamlit_mode=False):
    """
    Realiza un modelado de regresión lineal simple para la calidad del aire
    (PM10) en un departamento seleccionado.

    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
                                Se espera que la columna 'anio' sea de tipo datetime.
        streamlit_mode (bool): Si es True, muestra la salida en Streamlit.
                               Si es False, usa print y plt.show().
    """
    st.subheader("Modelado de Calidad del Aire (Regresión Lineal)")

    # Obtener lista de departamentos y contaminantes únicos del DataFrame
    departamentos_disponibles = df_aire['departamento'].unique().tolist()
    contaminantes_disponibles = df_aire['variable'].unique().tolist()

    '''
    # Filtros interactivos para el usuario
    departamento_objetivo = st.selectbox(
        "Selecciona el departamento para el aire:",
        departamentos_disponibles,
        index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
        key='air_dept_select'
    )
    contaminante_objetivo = st.selectbox(
        "Selecciona el contaminante a modelar:",
        contaminantes_disponibles,
        index=contaminantes_disponibles.index('PM10') if 'PM10' in contaminantes_disponibles else 0,
        key='air_cont_select'
    )
    '''

    # Usar st.columns para colocar los selectbox uno al lado del otro
    col1, col2 = st.columns(2)

    with col1:
        departamento_objetivo = st.selectbox(
            "Selecciona el departamento:",
            departamentos_disponibles,
            index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
            key='adv_air_dept_select'
        )
    with col2:
        contaminante_objetivo = st.selectbox(
            "Selecciona el contaminante a modelar:",
            contaminantes_disponibles,
            index=contaminantes_disponibles.index('PM10') if 'PM10' in contaminantes_disponibles else 0,
            key='adv_air_cont_select'
        )


    # Filtrar datos según las selecciones del usuario
    df_filtrado = df_aire[
        (df_aire['departamento'] == departamento_objetivo) &
        (df_aire['variable'] == contaminante_objetivo)
    ].copy() # Usar .copy() para evitar SettingWithCopyWarning

    if df_filtrado.empty:
        mensaje = f"No hay datos de {contaminante_objetivo} en {departamento_objetivo} para modelar."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    # Asegurarse de que 'promedio' no contenga NaNs para el modelado
    df_filtrado.dropna(subset=['promedio'], inplace=True)
    if df_filtrado.empty:
        mensaje = f"No hay suficientes datos limpios de {contaminante_objetivo} en {departamento_objetivo} para modelar después de eliminar NaNs."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    # Usar el año numérico como característica (X)
    X = df_filtrado[['anio']].apply(lambda x: x.dt.year).values # Convertir datetime a año numérico
    y = df_filtrado['promedio'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred_test = modelo.predict(X_test)

    # Calcular métricas de evaluación en el conjunto de prueba
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    if streamlit_mode:
        st.markdown(f"**Resultados del Modelo (Regresión Lineal) para {contaminante_objetivo} en {departamento_objetivo}:**")
        st.write(f"**Coeficiente (cambio por año):** {modelo.coef_[0]:.4f}")
        st.write(f"**Término independiente (valor en año 0):** {modelo.intercept_:.4f}")
        st.write(f"**RMSE (Error Cuadrático Medio):** {rmse:.4f}")
        st.write(f"**R² (Coeficiente de Determinación):** {r2:.4f}")

        # Predicciones sobre el conjunto completo de datos para la visualización
        y_full_pred = modelo.predict(X)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción de {contaminante_objetivo} en {departamento_objetivo}")
        ax.set_xlabel("Año")
        
        # Obtener la unidad correcta del diccionario
        unidad = UNIDADES_CONTAMINANTES.get(contaminante_objetivo, 'Unidad Desconocida')
        ax.set_ylabel(f"{contaminante_objetivo} Promedio ({unidad})")
        
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) # CERRAR LA FIGURA
    else:
        print(f"\nResultados del Modelo (Regresión Lineal) para {contaminante_objetivo} en {departamento_objetivo}:")
        print(f"Coeficiente: {modelo.coef_[0]:.4f}")
        print(f"Término independiente: {modelo.intercept_:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

        y_full_pred = modelo.predict(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción de {contaminante_objetivo} en {departamento_objetivo}")
        ax.set_xlabel("Año")
        unidad = UNIDADES_CONTAMINANTES.get(contaminante_objetivo, 'Unidad Desconocida')
        ax.set_ylabel(f"{contaminante_objetivo} Promedio ({unidad})")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig) # CERRAR LA FIGURA

def modelar_calidad_agua(df_agua, streamlit_mode=False):
    """
    Realiza un modelado de regresión lineal simple para la calidad del agua
    (IRCA) en un departamento seleccionado.

    Args:
        df_agua (pd.DataFrame): DataFrame con los datos de calidad del agua.
                                Se espera que la columna 'anio' sea de tipo datetime.
        streamlit_mode (bool): Si es True, muestra la salida en Streamlit.
                               Si es False, usa print y plt.show().
    """
    st.subheader("Modelado de Calidad del Agua (Regresión Lineal)")

    # Obtener lista de departamentos únicos del DataFrame
    departamentos_disponibles = df_agua['departamento'].unique().tolist()

    # Filtro interactivo para el usuario
    departamento_objetivo = st.selectbox(
        "Selecciona el departamento para el agua:",
        departamentos_disponibles,
        index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
        key='water_dept_select'
    )

    # Filtrar datos y agrupar por año para obtener el IRCA promedio
    df_filtrado = df_agua[df_agua['departamento'] == departamento_objetivo].copy()
    df_grouped = df_filtrado.groupby(df_filtrado['anio'].dt.year)['irca'].mean().reset_index()
    df_grouped.rename(columns={'anio': 'anio_num'}, inplace=True) # Renombrar para claridad

    if df_grouped.empty:
        mensaje = f"No hay datos de IRCA en {departamento_objetivo} para modelar."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    # Asegurarse de que 'irca' no contenga NaNs para el modelado
    df_grouped.dropna(subset=['irca'], inplace=True)
    if df_grouped.empty:
        mensaje = f"No hay suficientes datos limpios de IRCA en {departamento_objetivo} para modelar después de eliminar NaNs."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    X = df_grouped[['anio_num']].values # Usar el año numérico
    y = df_grouped['irca'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de Regresión Lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred_test = modelo.predict(X_test)

    # Calcular métricas de evaluación en el conjunto de prueba
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    if streamlit_mode:
        st.markdown(f"**Resultados del Modelo (Regresión Lineal) para IRCA en {departamento_objetivo}:**")
        st.write(f"**Coeficiente (cambio por año):** {modelo.coef_[0]:.4f}")
        st.write(f"**Término independiente (valor en año 0):** {modelo.intercept_:.4f}")
        st.write(f"**RMSE (Error Cuadrático Medio):** {rmse:.4f}")
        st.write(f"**R² (Coeficiente de Determinación):** {r2:.4f}")

        # Predicciones sobre el conjunto completo de datos para la visualización
        y_full_pred = modelo.predict(X)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='green', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción del IRCA Promedio en {departamento_objetivo}")
        ax.set_xlabel("Año")
        ax.set_ylabel("IRCA Promedio (%)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) # CERRAR LA FIGURA
    else:
        print(f"\nResultados del Modelo (Regresión Lineal) para IRCA en {departamento_objetivo}:")
        print(f"Coeficiente: {modelo.coef_[0]:.4f}")
        print(f"Término independiente: {modelo.intercept_:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

        y_full_pred = modelo.predict(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='green', label='Datos reales')
        ax.plot(X, y_full_pred, color='red', label='Predicción (Regresión Lineal)')
        ax.set_title(f"Predicción del IRCA Promedio en {departamento_objetivo}")
        ax.set_xlabel("Año")
        ax.set_ylabel("IRCA Promedio (%)")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig) # CERRAR LA FIGURA

# Ejecución directa para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    import src.loader as loader # Importar loader correctamente desde src
    df_aire_test, df_agua_test = loader.cargar_datos() # Cargar datos para la prueba
    if df_aire_test is not None and df_agua_test is not None:
        modelar_calidad_aire(df_aire_test)
        modelar_calidad_agua(df_agua_test)
    else:
        print("No se pudieron cargar los datos para ejecutar el modelado.")