import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from streamlit_extras.metric_cards import style_metric_cards
# loader no se importa aquí porque los DataFrames se pasarán como argumentos

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

def clasificar_irca(valor):
    """
    Clasifica el valor del IRCA en categorías de riesgo según la proporción de puntaje:
    0 a 5 %: Sin riesgo
    5,1 a 14 %: Riesgo bajo
    14,1 a 35 %: Riesgo medio
    35,1 a 80 %: Riesgo alto
    80,1 a 100 %: Inviable sanitariamente
    """
    if pd.isna(valor):
        return "Sin datos"
    if 0 <= valor <= 5:
        return "Sin riesgo"
    elif 5 < valor <= 14: # Cubre 5.1 a 14
        return "Riesgo bajo"
    elif 14 < valor <= 35: # Cubre 14.1 a 35
        return "Riesgo medio"
    elif 35 < valor <= 80: # Cubre 35.1 a 80
        return "Riesgo alto"
    elif 80 < valor <= 100: # Cubre 80.1 a 100
        return "Riesgo inviable sanitariamente"
    else:
        return "Valor fuera de rango"

def clasificar_pm10(valor):
    """Clasifica el valor de PM10 (promedio 24 horas) según la Resolución 2254 de 2017."""
    if pd.isna(valor):
        return "Sin datos"
    
    if 0 <= valor <= 50:
        return "Buena" # La contaminación atmosférica supone un riesgo bajo o nulo para la salud.
    elif 50 < valor <= 100:
        return "Aceptable" # Algunas personas sensibles pueden experimentar efectos adversos leves.
    elif 100 < valor <= 150:
        return "Dañina a la salud de grupos sensibles" # Niños, ancianos y aquellos con enfermedades respiratorias pueden verse afectados.
    elif 150 < valor <= 200:
        return "Dañina a la salud" # Toda la población puede comenzar a experimentar efectos adversos para la salud.
    elif 200 < valor <= 300:
        return "Muy dañina a la salud" # Riesgo significativo de efectos adversos para la salud en toda la población.
    elif valor > 300: # La resolución menciona hasta 500 para Peligrosa, pero el rango ICA es 301-500.
        return "Peligrosa" # Riesgo grave de efectos adversos para la salud en toda la población.
    else:
        return "Valor fuera de rango"

def clasificar_pm25(valor):
    """Clasifica el valor de PM2.5 (promedio 24 horas) según la Resolución 2254 de 2017."""
    if pd.isna(valor):
        return "Sin datos"
    
    if 0 <= valor <= 12:
        return "Buena" # La contaminación atmosférica supone un riesgo bajo o nulo para la salud.
    elif 12 < valor <= 37:
        return "Aceptable" # Algunas personas sensibles pueden experimentar efectos adversos leves.
    elif 37 < valor <= 55:
        return "Dañina a la salud de grupos sensibles" # Niños, ancianos y aquellos con enfermedades respiratorias pueden verse afectados.
    elif 55 < valor <= 150: # La resolución tiene 56-150 para dañina a la salud.
        return "Dañina a la salud" # Toda la población puede comenzar a experimentar efectos adversos para la salud.
    elif 150 < valor <= 250:
        return "Muy dañina a la salud" # Riesgo significativo de efectos adversos para la salud en toda la población.
    elif valor > 250: # La resolución menciona hasta 500 para Peligrosa, pero el rango ICA es 251-500.
        return "Peligrosa" # Riesgo grave de efectos adversos para la salud en toda la población.
    else:
        return "Valor fuera de rango"

def mostrar_dashboard(df_aire, df_agua):
    """
    Muestra el dashboard interactivo de calidad del aire y agua.
    
    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
        df_agua (pd.DataFrame): DataFrame con los datos de calidad del agua.
    """
    # ===============================================
    # Estilo CSS fondo negro elegante
    # ===============================================
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000;
            font-family: 'Segoe UI', sans-serif;
        }
        .stMetricValue {
            font-size: 24px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ===============================================
    # Encabezado
    # ===============================================
    col1, col2 = st.columns([7, 3])
    with col1:
        st.title("Dashboard de Calidad del Aire y Agua en Colombia")
        st.caption("Departamentos: Córdoba, Cesar y Bolívar")
    with col2:
        fecha_actual = datetime.datetime.now().strftime('%d %b %Y, %H:%M')
        st.caption("")
        st.success(f"Última actualización: {fecha_actual}")

    st.markdown("---")

    # ===============================================
    # Filtros
    # ===============================================
    departamentos_interes = ['CÓRDOBA', 'CESAR', 'BOLÍVAR']

    col_filtros1, col_filtros2 = st.columns(2)

    with col_filtros1:
        departamento = st.selectbox("Selecciona el departamento:", departamentos_interes)

    min_anio = df_aire['anio'].dt.year.min() if not df_aire.empty else 2000
    max_anio = df_aire['anio'].dt.year.max() if not df_aire.empty else datetime.datetime.now().year

    with col_filtros2:
        rango_anios = st.slider(
            "Selecciona rango de años:",
            min_value=int(min_anio),
            max_value=int(max_anio),
            value=(int(min_anio), int(max_anio)),
            step=1
        )

    # ===============================================
    # Filtrado de datos basado en las selecciones del usuario
    # ===============================================
    df_aire_filtrado = df_aire[
        (df_aire['departamento'] == departamento) &
        (df_aire['anio'].dt.year >= rango_anios[0]) &
        (df_aire['anio'].dt.year <= rango_anios[1])
    ].copy()

    df_agua_filtrado = df_agua[
        (df_agua['departamento'] == departamento) &
        (df_agua['anio'].dt.year >= rango_anios[0]) &
        (df_agua['anio'].dt.year <= rango_anios[1])
    ].copy()

    # ===============================================
    # KPIs (Indicadores Clave de Desempeño)
    # ===============================================
    st.subheader("Indicadores Clave de Desempeño (KPIs)")

    col1_kpi, col2_kpi, col3_kpi = st.columns(3)

    # PM10
    promedio_pm10 = df_aire_filtrado[df_aire_filtrado['variable'].str.upper() == 'PM10']['promedio'].mean()
    if pd.isna(promedio_pm10):
        promedio_pm10_texto = "Sin datos"
    else:
        promedio_pm10_texto = f"{promedio_pm10:.2f} µg/m³"

    # PM2.5
    promedio_pm25 = df_aire_filtrado[df_aire_filtrado['variable'].str.upper() == 'PM2.5']['promedio'].mean()
    if pd.isna(promedio_pm25):
        promedio_pm25_texto = "Sin datos"
    else:
        promedio_pm25_texto = f"{promedio_pm25:.2f} µg/m³"

    # IRCA
    promedio_irca = df_agua_filtrado['irca'].mean()
    if pd.isna(promedio_irca):
        promedio_irca_texto = "Sin datos"
    else:
        promedio_irca_texto = f"{promedio_irca:.2f}"

    with col1_kpi:
        st.metric("PM10 Promedio", promedio_pm10_texto)
    with col2_kpi:
        st.metric("PM2.5 Promedio", promedio_pm25_texto)
    with col3_kpi:
        st.metric("IRCA Promedio", promedio_irca_texto)

    style_metric_cards(background_color="#000", border_left_color="#009999", border_color="#DDDDDD")
    st.markdown("---")

    # ===============================================
    # Visualizaciones con Plotly
    # ===============================================
    st.subheader("Visualizaciones de Datos")

    # --- CAMBIO CLAVE AQUÍ: Obtener variables dinámicamente ---
    # Obtener todas las variables únicas con datos no nulos para el departamento y rango de años seleccionados
    df_aire_multi_var_for_plot = df_aire_filtrado.dropna(subset=['promedio']).copy()

    if not df_aire_multi_var_for_plot.empty:
        st.markdown(f"**Tendencia de Contaminantes y Parámetros Atmosféricos en {departamento}**")
        
        df_aire_multi_var_grouped = df_aire_multi_var_for_plot.groupby([df_aire_multi_var_for_plot['anio'].dt.year, 'variable'])['promedio'].mean().reset_index()
        df_aire_multi_var_grouped.rename(columns={'anio': 'Año', 'promedio': 'Valor Promedio'}, inplace=True)

        fig_multi_aire = px.line(
            df_aire_multi_var_grouped,
            x='Año',
            y='Valor Promedio',
            color='variable',
            markers=True,
            labels={'Año': 'Año', 'Valor Promedio': 'Valor Promedio'},
            template='plotly_dark',
            title=f"Tendencia de Contaminantes y Parámetros en {departamento} ({rango_anios[0]}-{rango_anios[1]})"
        )
        
        # Mapear unidades para el hovertemplate. Se usa .fillna('Unidad Desconocida') por si alguna variable no está en el diccionario.
        fig_multi_aire.update_traces(
            hovertemplate="<b>%{data.name}</b><br>Año: %{x}<br>Valor: %{y:.2f} %{customdata[0]}",
            customdata=df_aire_multi_var_grouped['variable'].map(UNIDADES_CONTAMINANTES).fillna('Unidad Desconocida').values.reshape(-1, 1)
        )

        st.plotly_chart(fig_multi_aire, use_container_width=True)
    else:
        st.info("No hay datos de contaminantes de aire para este departamento en el rango seleccionado.")

    # Boxplot IRCA
    if not df_agua_filtrado.empty:
        st.markdown("**Distribución de IRCA por Año**")
        df_agua_filtrado['anio_num'] = df_agua_filtrado['anio'].dt.year 
        fig_irca = px.box(
            df_agua_filtrado,
            x='anio_num',
            y='irca',
            labels={'anio_num': 'Año', 'irca': 'IRCA'},
            template='plotly_dark',
            color_discrete_sequence=['#66b2b2'],
            title=f"Distribución de IRCA en {departamento}"
        )
        st.plotly_chart(fig_irca, use_container_width=True)
    else:
        st.info("No hay datos de IRCA para este departamento en el rango seleccionado.")

    st.markdown("---")

    # ===============================================
    # Tablas de datos
    # ===============================================
    st.subheader("Datos Filtrados")

    tab1, tab2 = st.tabs(["Calidad del Aire", "Calidad del Agua"])
    with tab1:
        st.dataframe(df_aire_filtrado, use_container_width=True)
    with tab2:
        st.dataframe(df_agua_filtrado, use_container_width=True)

    st.markdown("---")

    # ===============================================
    # Alertas
    # ===============================================
    st.subheader("Alertas")

    # Alerta PM10
    if not pd.isna(promedio_pm10):
        categoria_pm10 = clasificar_pm10(promedio_pm10)
        if categoria_pm10 == "Peligrosa":
            st.error(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. Riesgo grave de efectos adversos para la salud en toda la población.")
        elif categoria_pm10 == "Muy dañina a la salud":
            st.error(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. Riesgo significativo de efectos adversos para la salud en toda la población.")
        elif categoria_pm10 == "Dañina a la salud":
            st.error(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. Toda la población puede comenzar a experimentar efectos adversos para la salud.")
        elif categoria_pm10 == "Dañina a la salud de grupos sensibles":
            st.warning(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. Niños, ancianos y aquellos con enfermedades respiratorias pueden verse afectados.")
        elif categoria_pm10 == "Aceptable":
            st.info(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. Algunas personas sensibles pueden experimentar efectos adversos leves.")
        elif categoria_pm10 == "Buena":
            st.success(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire. La contaminación atmosférica supone un riesgo bajo o nulo para la salud.")
        else:
            st.info(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire.")

    # Alerta PM2.5
    if not pd.isna(promedio_pm25):
        categoria_pm25 = clasificar_pm25(promedio_pm25)
        if categoria_pm25 == "Peligrosa":
            st.error(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. Riesgo grave de efectos adversos para la salud en toda la población.")
        elif categoria_pm25 == "Muy dañina a la salud":
            st.error(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. Riesgo significativo de efectos adversos para la salud en toda la población.")
        elif categoria_pm25 == "Dañina a la salud":
            st.error(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. Toda la población puede comenzar a experimentar efectos adversos para la salud.")
        elif categoria_pm25 == "Dañina a la salud de grupos sensibles":
            st.warning(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. Niños, ancianos y aquellos con enfermedades respiratorias pueden verse afectados.")
        elif categoria_pm25 == "Aceptable":
            st.info(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. Algunas personas sensibles pueden experimentar efectos adversos leves.")
        elif categoria_pm25 == "Buena":
            st.success(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire. La contaminación atmosférica supone un riesgo bajo o nulo para la salud.")
        else:
            st.info(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire.")

    # Alerta IRCA
    if not pd.isna(promedio_irca):
        categoria_irca = clasificar_irca(promedio_irca)
        if categoria_irca == "Sin riesgo":
            st.success("El agua en este departamento se encuentra sin riesgo según el IRCA, indicando buena calidad sanitaria.")
        elif categoria_irca == "Riesgo bajo":
            st.info("El IRCA indica un riesgo bajo, recomendable mantener monitoreo continuo para conservar la calidad.")
        elif categoria_irca == "Riesgo medio":
            st.warning("El IRCA indica riesgo medio. Se recomienda investigar y mejorar los procesos de tratamiento de agua.")
        elif categoria_irca == "Riesgo alto":
            st.error("El IRCA indica riesgo alto en la calidad del agua. Urge intervención para mejorar el tratamiento de agua y reducir riesgos sanitarios.")
        elif categoria_irca == "Riesgo inviable sanitariamente":
            st.error("El IRCA indica un nivel inviable sanitariamente. Se recomienda no consumir el agua hasta que se realicen acciones de remediación.")

    st.markdown("---")

    # ===============================================
    # Insights
    # ===============================================
    st.subheader("Insights")
    
    # Insight PM10
    if not pd.isna(promedio_pm10):
        categoria_pm10 = clasificar_pm10(promedio_pm10)
        if categoria_pm10 == "Peligrosa":
            st.error("El nivel de PM10 es **Peligroso**. Se requiere acción inmediata para proteger la salud pública debido al riesgo grave de efectos adversos en toda la población.")
        elif categoria_pm10 == "Muy dañina a la salud":
            st.error("La calidad del aire es **Muy dañina a la salud** por niveles elevados de PM10. Se recomienda implementar medidas urgentes de reducción de emisiones y evitar la exposición al aire libre.")
        elif categoria_pm10 == "Dañina a la salud":
            st.error("La calidad del aire es **Dañina a la salud** por niveles de PM10. Toda la población puede experimentar efectos adversos. Se aconseja reducir la exposición.")
        elif categoria_pm10 == "Dañina a la salud de grupos sensibles":
            st.warning("La calidad del aire es **Dañina a la salud de grupos sensibles** por PM10. Niños, ancianos y personas con enfermedades respiratorias deben limitar la exposición al aire libre.")
        elif categoria_pm10 == "Aceptable":
            st.info("La calidad del aire es **Aceptable** en términos de PM10. Algunas personas sensibles pueden experimentar efectos leves. Se recomienda vigilancia.")
        elif categoria_pm10 == "Buena":
            st.success("La calidad del aire es **Buena** en términos de PM10. La contaminación atmosférica supone un riesgo bajo o nulo para la salud.")
        else:
            st.info(f"El PM10 promedio ({promedio_pm10:.2f} µg/m³) indica **{categoria_pm10}** calidad del aire.")

    # Insight PM2.5
    if not pd.isna(promedio_pm25):
        categoria_pm25 = clasificar_pm25(promedio_pm25)
        if categoria_pm25 == "Peligrosa":
            st.error("El nivel de PM2.5 es **Peligroso**. Riesgo grave de efectos adversos para la salud en toda la población. Se requiere acción inmediata.")
        elif categoria_pm25 == "Muy dañina a la salud":
            st.error("La calidad del aire es **Muy dañina a la salud** por niveles elevados de PM2.5. Riesgo significativo de efectos adversos para la salud en toda la población. Evitar la exposición al aire libre.")
        elif categoria_pm25 == "Dañina a la salud":
            st.error("La calidad del aire es **Dañina a la salud** por niveles de PM2.5. Toda la población puede experimentar efectos adversos. Se aconseja reducir la exposición.")
        elif categoria_pm25 == "Dañina a la salud de grupos sensibles":
            st.warning("La calidad del aire es **Dañina a la salud de grupos sensibles** por PM2.5. Personas sensibles deben limitar la exposición al aire libre. Puede causar irritación respiratoria.")
        elif categoria_pm25 == "Aceptable":
            st.info("La calidad del aire es **Aceptable** en términos de PM2.5. Algunas personas sensibles pueden experimentar efectos leves. Se recomienda vigilancia.")
        elif categoria_pm25 == "Buena":
            st.success("La calidad del aire es **Buena** en términos de PM2.5. La contaminación atmosférica supone un riesgo bajo o nulo para la salud.")
        else:
            st.info(f"El PM2.5 promedio ({promedio_pm25:.2f} µg/m³) indica **{categoria_pm25}** calidad del aire.")

    # Insight IRCA
    if not pd.isna(promedio_irca):
        categoria_irca = clasificar_irca(promedio_irca)
        if categoria_irca == "Sin riesgo":
            st.success("El agua en este departamento se encuentra sin riesgo según el IRCA, indicando buena calidad sanitaria.")
        elif categoria_irca == "Riesgo bajo":
            st.info("El IRCA indica un riesgo bajo, recomendable mantener monitoreo continuo para conservar la calidad.")
        elif categoria_irca == "Riesgo medio":
            st.warning("El IRCA indica riesgo medio. Se recomienda investigar y mejorar los procesos de tratamiento de agua.")
        elif categoria_irca == "Riesgo alto":
            st.error("El IRCA indica riesgo alto en la calidad del agua. Urge intervención para mejorar el tratamiento de agua y reducir riesgos sanitarios.")
        elif categoria_irca == "Riesgo inviable sanitariamente":
            st.error("El IRCA indica un nivel inviable sanitariamente. Se recomienda no consumir el agua hasta que se realicen acciones de remediación.")
