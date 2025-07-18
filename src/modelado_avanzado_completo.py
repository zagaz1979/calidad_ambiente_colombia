import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

# Diccionario para mapear contaminantes a sus unidades, basado en la documentación.
UNIDADES_CONTAMINANTES = {
    'PM10': 'µg/m³',    # Partículas con diámetro aerodinámico de 10 micrómetros o menos.
    'O3': 'µg/m³',      # Ozono troposférico.
    'PST': 'µg/m³',     # Partículas Suspendidas Totales.
    'P': 'hPa o mbar',  # Presión atmosférica.
    'PM2.5': 'µg/m³',   # Partículas con diámetro aerodinámico de 2.5 micrómetros o menos.
    'TAire2': '°C',     # Temperatura del aire.
    'SO2': 'µg/m³',     # Dióxido de azufre.
    'NO2': 'µg/m³',     # Dióxido de nitrógeno.
    'CO': 'ppm o mg/m³',# Monóxido de carbono.
    'HAire2': '%',      # Humedad relativa del aire.
    'DViento': 'grados',# Dirección del viento.
    'RGlobal': 'W/m²',  # Radiación solar global.
    'VViento': 'm/s o km/h', # Velocidad del viento.
    'PLiquida': 'mm'    # Precipitación líquida (lluvia, llovizna, etc.).
}


def modelar_avanzado_completo(df_aire):
    """
    Realiza un modelado predictivo avanzado de la calidad del aire
    comparando Regresión Lineal, Regresión Polinómica y Random Forest.
    Permite la selección interactiva de departamento y contaminante.

    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
                                Se espera que la columna 'anio' sea de tipo datetime.
    """
    st.title("Modelado Avanzado de Calidad del Aire")
    st.write("Compara el rendimiento de Regresión Lineal, Regresión Polinómica y Random Forest para la predicción de contaminantes.")

    # Obtener lista de departamentos únicos del DataFrame
    departamentos_disponibles = df_aire['departamento'].unique().tolist()
    
    # Usar st.columns para colocar los selectbox uno al lado del otro
    col1, col2 = st.columns(2)

    with col1:
        departamento_objetivo = st.selectbox(
            "Selecciona el departamento:",
            departamentos_disponibles,
            index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
            key='adv_air_dept_select_comp' # Clave única para este selectbox
        )
    
    # --- FILTRADO DINÁMICO DE CONTAMINANTES DISPONIBLES POR DEPARTAMENTO ---
    # Filtrar el DataFrame por el departamento seleccionado para obtener sus contaminantes
    df_aire_por_departamento = df_aire[df_aire['departamento'] == departamento_objetivo].copy()
    
    # Obtener solo las variables que tienen datos no nulos en 'promedio' para el departamento seleccionado
    # Esto asegura que solo se muestren contaminantes con datos válidos para modelar
    contaminantes_disponibles_por_dep = df_aire_por_departamento.dropna(subset=['promedio'])['variable'].unique().tolist()
    
    # Si no hay contaminantes disponibles para el departamento seleccionado, mostrar advertencia y salir
    if not contaminantes_disponibles_por_dep:
        st.warning(f"No hay contaminantes con datos disponibles para '{departamento_objetivo}'. Por favor, selecciona otro departamento.")
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
            key='adv_air_cont_select_comp' # Clave única para este selectbox
        )
    # --- FIN FILTRADO DINÁMICO ---
    
    # Filtrar datos según las selecciones del usuario (ahora contaminante_objetivo ya es válido para el departamento)
    df_filtrado = df_aire_por_departamento[
        (df_aire_por_departamento['variable'] == contaminante_objetivo)
    ].copy()

    # Asegurarse de que 'promedio' y 'anio' no contengan NaNs para el modelado
    df_filtrado.dropna(subset=['anio', 'promedio'], inplace=True)
    if df_filtrado.empty:
        st.warning(f"No hay suficientes datos limpios de {contaminante_objetivo} en {departamento_objetivo} para modelar después de eliminar NaNs. Esto no debería ocurrir si la lista de selección se filtró correctamente.")
        return

    # Usar el año numérico como característica (X)
    X = df_filtrado[['anio']].apply(lambda x: x.dt.year).values # Convertir datetime a año numérico
    y = df_filtrado['promedio'].values

    # ======================================================================
    # Verificaciones para train_test_split y cálculo de métricas
    # ======================================================================
    
    # 1. Verificar si hay suficientes puntos de datos para el split
    if len(X) < 2: # Necesitas al menos 2 puntos para cualquier split
        st.warning(f"No hay suficientes puntos de datos ({len(X)}) para realizar el modelado y la evaluación. Se necesitan al menos 2 puntos de datos limpios.")
        # Solo mostrar el mensaje y salir, sin métricas ni resultados
        return

    # Determinar test_size dinámicamente para asegurar al menos 1 muestra de prueba
    if len(X) < 5: # Si el total de puntos de datos es pequeño
        test_size_val = 1 / len(X) if len(X) > 1 else 0 # Asegura al menos 1 en test si >1 total
        st.info(f"Advertencia: Pocos datos ({len(X)}). El conjunto de prueba será muy pequeño. Las métricas pueden no ser representativas.")
    else:
        test_size_val = 0.2

    # Realizar el split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

    # 2. Verificar el conjunto de prueba (y_test) para el cálculo de métricas
    if len(y_test) < 2 or y_test.var() == 0:
        st.warning(f"No hay suficientes datos en el conjunto de prueba ({len(y_test)} puntos) o la varianza de los valores reales es cero. Las métricas de R² serán 'nan' y RMSE podría ser '0.0000' (no representativo).")
        # Solo mostrar el mensaje y salir, sin métricas ni resultados
        return # Salir de la función aquí si las métricas no son significativas


    # =========================
    # Regresión Lineal
    # =========================
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)
    y_pred_lineal_test = modelo_lineal.predict(X_test) # Predicción en test set

    # =========================
    # Regresión Polinómica (grado 2)
    # =========================
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test) # Transformar el test set
    
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly_train, y_train)
    y_pred_poly_test = modelo_poly.predict(X_poly_test) # Predicción en test set

    # =========================
    # Random Forest
    # =========================
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf_test = modelo_rf.predict(X_test) # Predicción en test set

    # =========================
    # Métricas (evaluadas en el conjunto de prueba)
    # =========================
    r2_lineal = r2_score(y_test, y_pred_lineal_test)
    rmse_lineal = np.sqrt(mean_squared_error(y_test, y_pred_lineal_test))

    r2_poly = r2_score(y_test, y_pred_poly_test)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly_test))

    r2_rf = r2_score(y_test, y_pred_rf_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

    # =========================
    # Predicciones para visualización (sobre todos los datos históricos)
    # =========================
    y_pred_lineal_full = modelo_lineal.predict(X)
    y_pred_poly_full = modelo_poly.predict(poly.transform(X))
    y_pred_rf_full = modelo_rf.predict(X)

    # =========================
    # Predicciones 2024-2025
    # =========================
    st.subheader("Predicciones para 2024 y 2025")
    # Generar años futuros
    max_anio_historico = X.max() 
    X_futuro_years = np.array([[year] for year in range(max_anio_historico + 1, max_anio_historico + 3)])
    
    # Transformar para modelo polinómico
    X_futuro_poly = poly.transform(X_futuro_years)
    
    # Realizar predicciones futuras
    pred_lineal_futuro = modelo_lineal.predict(X_futuro_years)
    pred_poly_futuro = modelo_poly.predict(X_futuro_poly)
    pred_rf_futuro = modelo_rf.predict(X_futuro_years)

    # Crear DataFrame de predicciones futuras
    pred_df = pd.DataFrame({
        "Año": [year[0] for year in X_futuro_years],
        "Lineal": pred_lineal_futuro,
        "Polinómica": pred_poly_futuro,
        "Random Forest": pred_rf_futuro
    })
    st.table(pred_df)

    # =========================
    # Visualización Comparativa
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Datos reales')
    ax.plot(X, y_pred_lineal_full, color='red', linestyle='--', label='Lineal')
    ax.plot(X, y_pred_poly_full, color='green', linestyle='-.', label='Polinómica (grado 2)')
    ax.plot(X, y_pred_rf_full, color='purple', linestyle='-', label='Random Forest')
    ax.set_title(f"Comparación de Modelos de Predicción de {contaminante_objetivo} en {departamento_objetivo}")
    ax.set_xlabel("Año")
    
    # Obtener la unidad correcta del diccionario
    unidad = UNIDADES_CONTAMINANTES.get(contaminante_objetivo, 'Unidad Desconocida')
    ax.set_ylabel(f"{contaminante_objetivo} Promedio ({unidad})")
    
    ax.legend()
    st.pyplot(fig)
    # Guardar la figura antes de cerrarla
    try:
        output_dir = "./outputs"
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"comparacion_modelos_{contaminante_objetivo.lower()}_{departamento_objetivo.lower()}.png"))
        st.info(f"Gráfico guardado en: {output_dir}/comparacion_modelos_{contaminante_objetivo.lower()}_{departamento_objetivo.lower()}.png")
    except Exception as e:
        st.warning(f"No se pudo guardar el gráfico: {e}")
    plt.close(fig) # CERRAR LA FIGURA

    # =========================
    # Resultados en Streamlit
    # =========================
    st.subheader("Métricas de Evaluación (en datos de prueba)")

    col1_res, col2_res, col3_res = st.columns(3)
    with col1_res:
        st.metric("Lineal R²", f"{r2_lineal:.4f}" if not np.isnan(r2_lineal) else "nan")
        st.metric("Lineal RMSE", f"{rmse_lineal:.4f}")
    with col2_res:
        st.metric("Polinómica R²", f"{r2_poly:.4f}" if not np.isnan(r2_poly) else "nan")
    with col2_res:
        st.metric("Polinómica RMSE", f"{rmse_poly:.4f}")
    with col3_res:
        st.metric("Random Forest R²", f"{r2_rf:.4f}" if not np.isnan(r2_rf) else "nan")
    with col3_res:
        st.metric("Random Forest RMSE", f"{rmse_rf:.4f}")

# Ejecución directa opcional para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    import src.loader as loader # Importar loader correctamente desde src
    df_aire_test, _ = loader.cargar_datos() # Cargar datos para la prueba
    if df_aire_test is not None:
        modelar_avanzado_completo(df_aire_test)
    else:
        print("No se pudieron cargar los datos para ejecutar el modelado avanzado completo.")