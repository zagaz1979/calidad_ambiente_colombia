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
    'VViento': 'm/s o km/h', # Velocidad del viento. Velocidad con la que se mueve el aire.
                            # Puede influir en la dispersión de contaminantes.
    'PLiquida': 'mm'    # Precipitación líquida (lluvia, llovizna, etc.).
}


def modelar_calidad_aire(df_aire, streamlit_mode=False):
    """
    Realiza un modelado de regresión lineal simple para la calidad del aire
    en un departamento y contaminante seleccionados.

    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
                                Se espera que la columna 'anio' sea de tipo datetime.
        streamlit_mode (bool): Si es True, muestra la salida en Streamlit.
                               Si es False, usa print y plt.show().
    """
    st.subheader("Modelado de Calidad del Aire (Regresión Lineal)")

    # Obtener lista de departamentos únicos del DataFrame
    departamentos_disponibles = df_aire['departamento'].unique().tolist()
    
    # Usar st.columns para colocar los selectbox uno al lado del otro
    col1, col2 = st.columns(2)

    with col1:
        departamento_objetivo = st.selectbox(
            "Selecciona el departamento:",
            departamentos_disponibles,
            index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
            key='adv_air_dept_select_model'
        )
    
    # --- FILTRADO DINÁMICO DE CONTAMINANTES DISPONIBLES POR DEPARTAMENTO ---
    # Filtrar el DataFrame por el departamento seleccionado para obtener sus contaminantes
    df_aire_por_departamento = df_aire[df_aire['departamento'] == departamento_objetivo].copy()
    
    # Obtener solo las variables que tienen datos no nulos en 'promedio' para el departamento seleccionado
    # Esto asegura que solo se muestren contaminantes con datos válidos para modelar
    contaminantes_disponibles_por_dep = df_aire_por_departamento.dropna(subset=['promedio'])['variable'].unique().tolist()
    
    # Si no hay contaminantes disponibles para el departamento seleccionado, mostrar advertencia y salir
    if not contaminantes_disponibles_por_dep:
        mensaje = f"No hay contaminantes con datos disponibles para '{departamento_objetivo}'. Por favor, selecciona otro departamento."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return # Salir de la función si no hay datos de contaminantes

    # Asegurarse de que 'PM10' sea la opción predeterminada si está disponible, de lo contrario, la primera disponible
    default_contaminante_index = 0
    if 'PM10' in contaminantes_disponibles_por_dep:
        default_contaminante_index = contaminantes_disponibles_por_dep.index('PM10')
    elif contaminantes_disponibles_por_dep: # Si hay otros contaminantes, selecciona el primero
        default_contaminante_index = 0
    else: # Si la lista está vacía, esto ya se manejó arriba. Como fallback, poner -1 para que no seleccione nada si la lista es vacía.
        default_contaminante_index = -1 # No debería llegar aquí si la verificación de 'if not contaminantes_disponibles_por_dep' funciona.

    with col2:
        contaminante_objetivo = st.selectbox(
            "Selecciona el contaminante a modelar:",
            contaminantes_disponibles_por_dep, # Usar la lista filtrada dinámicamente
            index=default_contaminante_index,
            key='adv_air_cont_select_model'
        )

    # Filtrar datos según las selecciones del usuario (ahora contaminante_objetivo ya es válido para el departamento)
    df_filtrado = df_aire_por_departamento[
        (df_aire_por_departamento['variable'] == contaminante_objetivo)
    ].copy()

    # Asegurarse de que 'promedio' no contenga NaNs para el modelado
    df_filtrado.dropna(subset=['promedio'], inplace=True)
    if df_filtrado.empty:
        mensaje = f"No hay suficientes datos limpios de {contaminante_objetivo} en {departamento_objetivo} para modelar después de eliminar NaNs. Esto no debería ocurrir si la lista de selección se filtró correctamente."
        if streamlit_mode:
            st.warning(mensaje)
        else:
            print(mensaje)
        return

    # Usar el año numérico como característica (X)
    X = df_filtrado[['anio']].apply(lambda x: x.dt.year).values # Convertir datetime a año numérico
    y = df_filtrado['promedio'].values

    # ======================================================================
    # Verificaciones para train_test_split y cálculo de métricas
    # ======================================================================
    
    # 1. Verificar si hay suficientes puntos de datos para el split
    if len(X) < 2: # Necesitas al menos 2 puntos para cualquier split (train y test)
        mensaje = f"No hay suficientes puntos de datos ({len(X)}) para realizar el modelado y la evaluación de {contaminante_objetivo} en {departamento_objetivo}. Se necesitan al menos 2 puntos de datos limpios."
        if streamlit_mode:
            st.warning(mensaje)
            # No mostrar resultados ni predicciones, solo el mensaje
        else:
            print(mensaje)
        return

    # Determinar test_size dinámicamente para asegurar al menos 1 muestra de prueba
    if len(X) < 5: # Si el total de puntos de datos es pequeño
        test_size_val = 1 / len(X) if len(X) > 1 else 0 # Asegura al menos 1 en test si >1 total
        if streamlit_mode:
            st.info(f"Advertencia: Pocos datos ({len(X)}). El conjunto de prueba será muy pequeño. Las métricas pueden no ser representativas.")
    else:
        test_size_val = 0.2

    # Realizar el split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

    # 2. Verificar el conjunto de prueba (y_test) para el cálculo de métricas
    if len(y_test) < 2 or y_test.var() == 0:
        mensaje = f"No hay suficientes datos en el conjunto de prueba ({len(y_test)} puntos) o la varianza de los valores reales es cero para {contaminante_objetivo} en {departamento_objetivo}. Las métricas de R² serán 'nan' y RMSE podría ser '0.0000' (no representativo)."
        if streamlit_mode:
            st.warning(mensaje)
            # No mostrar resultados ni predicciones, solo el mensaje
        else:
            print(mensaje)
        return # Salir de la función aquí si las métricas no son significativas

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
        
        col1_res, col2_res = st.columns(2)
        with col1_res:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2_res:
            st.metric("R²", f"{r2:.4f}" if not np.isnan(r2) else "nan")


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
        key='water_dept_select_model'
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

    # ======================================================================
    # Verificaciones para train_test_split y cálculo de métricas (replicadas)
    # ======================================================================
    
    # 1. Verificar si hay suficientes puntos de datos para el split
    if len(X) < 2: # Necesitas al menos 2 puntos para cualquier split (train y test)
        mensaje = f"No hay suficientes puntos de datos ({len(X)}) para realizar el modelado y la evaluación de IRCA en {departamento_objetivo}. Se necesitan al menos 2 puntos de datos limpios."
        if streamlit_mode:
            st.warning(mensaje)
            # No mostrar resultados ni predicciones, solo el mensaje
        else:
            print(mensaje)
        return

    # Determinar test_size dinámicamente para asegurar al menos 1 muestra de prueba
    if len(X) < 5: # Si el total de puntos de datos es pequeño
        test_size_val = 1 / len(X) if len(X) > 1 else 0 # Asegura al menos 1 en test si >1 total
        if streamlit_mode:
            st.info(f"Advertencia: Pocos datos ({len(X)}). El conjunto de prueba será muy pequeño. Las métricas pueden no ser representativas.")
    else:
        test_size_val = 0.2

    # Realizar el split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

    # 2. Verificar el conjunto de prueba (y_test) para el cálculo de métricas
    if len(y_test) < 2 or y_test.var() == 0:
        mensaje = f"No hay suficientes datos en el conjunto de prueba ({len(y_test)} puntos) o la varianza de los valores reales es cero para IRCA en {departamento_objetivo}. Las métricas de R² serán 'nan' y RMSE podría ser '0.0000' (no representativo)."
        if streamlit_mode:
            st.warning(mensaje)
            # No mostrar resultados ni predicciones, solo el mensaje
        else:
            print(mensaje)
        return # Salir de la función aquí si las métricas no son significativas

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
        
        col1_res, col2_res = st.columns(2)
        with col1_res:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2_res:
            st.metric("R²", f"{r2:.4f}" if not np.isnan(r2) else "nan")

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
        modelar_calidad_aire(df_aire_test, streamlit_mode=True) # Ejecutar en modo Streamlit para ver el comportamiento
        modelar_calidad_agua(df_agua_test, streamlit_mode=True)
    else:
        print("No se pudieron cargar los datos para ejecutar el modelado.")
