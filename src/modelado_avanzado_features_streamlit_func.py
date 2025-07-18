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
from pathlib import Path
import os

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


def modelado_avanzado_features_streamlit(df_aire):
    """
    Realiza un modelado predictivo avanzado con selección interactiva de features
    para la calidad del aire, comparando Regresión Lineal, Polinómica y Random Forest.

    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
                                Se espera que la columna 'anio' sea de tipo datetime.
    """
    st.title("Modelado Predictivo Avanzado con Selección de Features")
    st.write("Selecciona las variables predictoras y compara el rendimiento de diferentes modelos.")

    # Obtener lista de departamentos únicos del DataFrame
    departamentos_disponibles = df_aire['departamento'].unique().tolist()
    
    # Usar st.columns para colocar los selectbox uno al lado del otro
    col1, col2 = st.columns(2)

    with col1:
        departamento_objetivo = st.selectbox(
            "Selecciona el departamento:",
            departamentos_disponibles,
            index=departamentos_disponibles.index('CÓRDOBA') if 'CÓRDOBA' in departamentos_disponibles else 0,
            key='feat_air_dept_select_func' # Clave única para este selectbox
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
            "Selecciona el contaminante a modelar (variable objetivo 'promedio'):",
            contaminantes_disponibles_por_dep, # Usar la lista filtrada dinámicamente
            index=default_contaminante_index,
            key='feat_air_cont_select_func' # Clave única para este selectbox
        )
    # --- FIN FILTRADO DINÁMICO ---

    # Filtrar el DataFrame por departamento y contaminante objetivo (ahora contaminante_objetivo ya es válido para el departamento)
    df_filtrado_por_seleccion = df_aire_por_departamento[
        (df_aire_por_departamento['variable'] == contaminante_objetivo)
    ].copy()

    if df_filtrado_por_seleccion.empty:
        st.warning(f"No hay datos disponibles para el contaminante '{contaminante_objetivo}' en el departamento de '{departamento_objetivo}' en el dataset actual. Esto no debería ocurrir si la lista de selección se filtró correctamente.")
        return

    # Opciones de variables numéricas para la selección de features
    df_filtrado_por_seleccion['anio_num'] = df_filtrado_por_seleccion['anio'].dt.year
    opciones_features = [
        'anio_num', 'dias_excedencias', 'percentil_98', 'maximo', 'minimo', 'mediana', 'suma', 'n_datos'
    ]
    
    opciones_features_existentes = [col for col in opciones_features if col in df_filtrado_por_seleccion.columns]

    st.subheader("Selección de Variables Predictoras")
    features_seleccionadas = st.multiselect(
        "Selecciona las variables que usarás para el modelado predictivo:",
        opciones_features_existentes,
        default=[f for f in ['anio_num', 'dias_excedencias', 'percentil_98', 'maximo', 'minimo'] if f in opciones_features_existentes]
    )

    if len(features_seleccionadas) == 0:
        st.warning("Selecciona al menos una variable predictora para continuar.")
        return

    # Definir X e y
    X_full = df_filtrado_por_seleccion[features_seleccionadas]
    y_full = df_filtrado_por_seleccion['promedio']

    # Imputación de valores faltantes (usando la mediana del conjunto completo)
    X_full = X_full.fillna(X_full.median())
    y_full = y_full.fillna(y_full.median())

    # Eliminar filas donde y_full sea NaN después de la imputación (si aún quedan)
    valid_indices = y_full.dropna().index
    X_full = X_full.loc[valid_indices]
    y_full = y_full.loc[valid_indices]

    if X_full.empty or y_full.empty:
        st.warning("No hay suficientes datos limpios para el modelado después de la imputación y eliminación de NaNs. Ajusta las selecciones.")
        return
    
    # ======================================================================
    # NUEVAS VERIFICACIONES PARA train_test_split y cálculo de métricas
    # ======================================================================
    
    # 1. Verificar si hay suficientes puntos de datos para el split
    if len(X_full) < 2: # Necesitas al menos 2 puntos para cualquier split
        st.warning(f"No hay suficientes puntos de datos ({len(X_full)}) para realizar el modelado y la evaluación. Se necesitan al menos 2 puntos de datos limpios.")
        # No mostrar resultados ni predicciones, solo el mensaje
        return

    # Determinar test_size dinámicamente para asegurar al menos 1 muestra de prueba
    # y evitar que el test set tenga varianza cero si es posible.
    # Si tenemos muy pocos datos, es mejor no hacer un split tradicional para R2.
    if len(X_full) < 5: # Si el total de puntos de datos es pequeño
        st.info(f"Advertencia: Pocos datos ({len(X_full)}). El conjunto de prueba será muy pequeño. Las métricas pueden no ser representativas.")
        test_size_val = 1 / len(X_full) if len(X_full) > 1 else 0 # Asegura al menos 1 en test si >1 total
    else:
        test_size_val = 0.2

    # Realizar el split
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size_val, random_state=42)

    # 2. Verificar el conjunto de prueba (y_test) para el cálculo de métricas
    # R2 es indefinido si la varianza de y_test es 0.
    # RMSE puede ser 0 si el modelo predice perfectamente un valor constante.
    if len(y_test) < 2 or y_test.var() == 0:
        st.warning(f"No hay suficientes datos en el conjunto de prueba ({len(y_test)} puntos) o la varianza de los valores reales es cero. Las métricas de R² serán 'nan' y RMSE podría ser '0.0000' (no representativo).")
        # No mostrar resultados ni predicciones, solo el mensaje
        return # Salir de la función aquí si las métricas no son significativas

    # =========================
    # Regresión Lineal
    # =========================
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, y_train)
    y_pred_lineal_test = modelo_lineal.predict(X_test)
    
    r2_lineal = r2_score(y_test, y_pred_lineal_test)
    rmse_lineal = np.sqrt(mean_squared_error(y_test, y_pred_lineal_test))

    # =========================
    # Regresión Polinómica (grado 2)
    # =========================
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly_train, y_train)
    y_pred_poly_test = modelo_poly.predict(X_poly_test)
    
    r2_poly = r2_score(y_test, y_pred_poly_test)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly_test))

    # =========================
    # Random Forest
    # =========================
    modelo_rf = RandomForestRegressor(random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf_test = modelo_rf.predict(X_test)
    
    r2_rf = r2_score(y_test, y_pred_rf_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

    st.subheader("Resultados de los Modelos (en datos de prueba)")
    col1, col2, col3 = st.columns(3)
    # Formatear R² para mostrar "nan" si es NaN
    col1.metric("Lineal R²", f"{r2_lineal:.4f}" if not np.isnan(r2_lineal) else "nan")
    col1.metric("Lineal RMSE", f"{rmse_lineal:.4f}")
    col2.metric("Polinómica R²", f"{r2_poly:.4f}" if not np.isnan(r2_poly) else "nan")
    col2.metric("Polinómica RMSE", f"{rmse_poly:.4f}")
    col3.metric("Random Forest R²", f"{r2_rf:.4f}" if not np.isnan(r2_rf) else "nan")
    col3.metric("Random Forest RMSE", f"{rmse_rf:.4f}")

    # =========================
    # Predicciones 2024-2025
    # =========================
    st.subheader("Predicciones para 2024 y 2025")
    
    if 'anio_num' in X_full.columns:
        max_anio_historico = X_full['anio_num'].max()
    else:
        # Fallback si 'anio_num' no está en X_full (aunque debería estar si se seleccionó)
        max_anio_historico = df_filtrado_por_seleccion['anio'].dt.year.max()

    futuro_anios = pd.DataFrame({'anio_num': [max_anio_historico + 1, max_anio_historico + 2]})
    
    median_train_features = X_train.median()
    for col in features_seleccionadas:
        if col != 'anio_num':
            futuro_anios[col] = median_train_features[col]

    futuro_anios_ordered = futuro_anios[[f for f in features_seleccionadas if f in futuro_anios.columns]]

    if 'anio_num' in features_seleccionadas:
        futuro_lineal = modelo_lineal.predict(futuro_anios_ordered)
        futuro_poly = modelo_poly.predict(poly.transform(futuro_anios_ordered))
        futuro_rf = modelo_rf.predict(futuro_anios_ordered)
    else:
        st.warning("Para predicciones futuras, 'anio_num' debe ser una de las variables seleccionadas. Las predicciones no se generarán.")
        st.dataframe(pd.DataFrame({'Año': futuro_anios['anio_num'], 'Lineal': ['N/A']*2, 'Polinómica': ['N/A']*2, 'Random Forest': ['N/A']*2}))
        return

    pred_df = pd.DataFrame({
        'Año': futuro_anios['anio_num'],
        'Lineal': futuro_lineal,
        'Polinómica': futuro_poly,
        'Random Forest': futuro_rf
    })
    st.dataframe(pred_df)

    # =========================
    # Gráficos y exportación
    # =========================
    Path('./outputs').mkdir(exist_ok=True) # Asegura que el directorio 'outputs' exista

    # Gráfico de dispersión de Valor Real vs. Predicción (en el conjunto de prueba)
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    ax_scatter.scatter(y_test, y_pred_lineal_test, label='Lineal', alpha=0.7)
    ax_scatter.scatter(y_test, y_pred_poly_test, label='Polinómica', alpha=0.7)
    ax_scatter.scatter(y_test, y_pred_rf_test, label='Random Forest', alpha=0.7)
    
    # Línea de 45 grados para referencia
    min_val = min(y_test.min(), y_pred_lineal_test.min(), y_pred_poly_test.min(), y_pred_rf_test.min())
    max_val = max(y_test.max(), y_pred_lineal_test.max(), y_pred_poly_test.max(), y_pred_rf_test.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Predicción Perfecta')
    
    ax_scatter.set_xlabel('Valor Real')
    ax_scatter.set_ylabel('Predicción')
    
    # Obtener la unidad correcta del diccionario para el título del gráfico
    unidad = UNIDADES_CONTAMINANTES.get(contaminante_objetivo, 'µg/m³')
    
    ax_scatter.set_title(f'Comparación de Modelos Predictivos ({contaminante_objetivo} en {departamento_objetivo}) - Unidades: {unidad}')
    
    ax_scatter.legend()
    plt.tight_layout()
    
    # Guardar la figura con el nombre correcto y dinámico
    output_filename = f'comparacion_modelos_features_{contaminante_objetivo.lower()}_{departamento_objetivo.lower()}.png'
    output_path_scatter = os.path.join('./outputs', output_filename)
    
    try:
        fig_scatter.savefig(output_path_scatter)
        st.success(f"Gráfico comparativo guardado automáticamente en {output_path_scatter}")
    except Exception as e:
        st.error(f"Error al guardar el gráfico comparativo: {e}")
    
    st.pyplot(fig_scatter) 

    plt.close(fig_scatter) # CERRAR LA FIGURA

    st.info("Proceso de modelado predictivo avanzado con selección de features completado.")

# Ejecución directa para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    import src.loader as loader # Importar loader correctamente desde src
    df_aire_test, _ = loader.cargar_datos() # Cargar datos para la prueba
    if df_aire_test is not None:
        modelado_avanzado_features_streamlit(df_aire_test)
    else:
        print("No se pudieron cargar los datos para ejecutar el modelado avanzado con features.")
