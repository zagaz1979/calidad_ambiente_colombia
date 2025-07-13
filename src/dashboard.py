import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from streamlit_extras.metric_cards import style_metric_cards
from src import loader

def clasificar_irca(valor):
    if pd.isna(valor):
        return "Sin datos"
    if 0 <= valor <= 5:
        return "Sin riesgo"
    elif 5 < valor <= 14:
        return "Riesgo bajo"
    elif 14 < valor <= 35:
        return "Riesgo medio"
    elif 35 < valor <= 80:
        return "Riesgo alto"
    elif 80 < valor <= 100:
        return "Riesgo inviable sanitariamente"
    else:
        return "Valor fuera de rango"

def clasificar_pm10(valor):
    if pd.isna(valor):
        return "Sin datos"
    if valor <= 15:
        return "Excelente"
    elif valor <= 25:
        return "Bueno"
    elif valor <= 35:
        return "Aceptable"
    elif valor <= 50:
        return "Regular"
    elif valor <= 75:
        return "Malo"
    else:
        return "Muy Malo"

def mostrar_dashboard():
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
        st.success(f"Última actualización: {fecha_actual}")

    st.markdown("---")

    # ===============================================
    # Filtros
    # ===============================================
    df_aire, df_agua = loader.cargar_datos(streamlit_mode=True)

    for df in [df_aire, df_agua]:
        df['anio'] = pd.to_datetime(df['anio'], format='%Y')

    departamentos_interes = ['CÓRDOBA', 'CESAR', 'BOLÍVAR']

    col_filtros1, col_filtros2 = st.columns(2)

    with col_filtros1:
        departamento = st.selectbox("Selecciona el departamento:", departamentos_interes)

    min_anio = df_aire['anio'].dt.year.min()
    max_anio = df_aire['anio'].dt.year.max()

    with col_filtros2:
        rango_anios = st.slider(
            "Selecciona rango de años:",
            min_value=min_anio,
            max_value=max_anio,
            value=(min_anio, max_anio),
            step=1
        )

    # ===============================================
    # Filtrado
    # ===============================================
    df_aire_filtrado = df_aire[
        (df_aire['departamento'] == departamento) &
        (df_aire['anio'].dt.year >= rango_anios[0]) &
        (df_aire['anio'].dt.year <= rango_anios[1])
    ]

    df_agua_filtrado = df_agua[
        (df_agua['departamento'] == departamento) &
        (df_agua['anio'].dt.year >= rango_anios[0]) &
        (df_agua['anio'].dt.year <= rango_anios[1])
    ]

    # ===============================================
    # KPIs
    # ===============================================
    st.subheader("Indicadores Clave de Desempeño (KPIs)")

    col1, col2, col3 = st.columns(3)

    # PM10
    promedio_pm10 = round(
        df_aire_filtrado[df_aire_filtrado['variable'].str.upper() == 'PM10']['promedio'].mean(),
        2
    )
    if pd.isna(promedio_pm10):
        promedio_pm10_texto = "Sin datos"
    else:
        promedio_pm10_texto = f"{promedio_pm10} µg/m³"

    # PM2.5
    promedio_pm25 = round(
        df_aire_filtrado[df_aire_filtrado['variable'].str.upper() == 'PM2.5']['promedio'].mean(),
        2
    )
    if pd.isna(promedio_pm25):
        promedio_pm25_texto = "Sin datos"
    else:
        promedio_pm25_texto = f"{promedio_pm25} µg/m³"

    # IRCA
    promedio_irca = round(
        df_agua_filtrado['irca'].mean(),
        2
    )
    if pd.isna(promedio_irca):
        promedio_irca_texto = "Sin datos"
    else:
        promedio_irca_texto = f"{promedio_irca}"

    with col1:
        st.metric("PM10 Promedio", promedio_pm10_texto)
    with col2:
        st.metric("PM2.5 Promedio", promedio_pm25_texto)
    with col3:
        st.metric("IRCA Promedio", promedio_irca_texto)

    style_metric_cards(background_color="#000", border_left_color="#009999", border_color="#DDDDDD")
    st.markdown("---")

    # ===============================================
    # Visualizaciones con Plotly
    # ===============================================
    st.subheader("Visualizaciones de Datos")

    # Tendencia PM10
    df_pm10 = df_aire_filtrado[df_aire_filtrado['variable'].str.upper() == 'PM10']
    if not df_pm10.empty:
        st.markdown("**Tendencia de PM10 (µg/m³) por Año**")
        df_pm10_grouped = df_pm10.groupby(df_pm10['anio'].dt.year)['promedio'].mean().reset_index()
        fig_pm10 = px.line(
            df_pm10_grouped,
            x='anio',
            y='promedio',
            markers=True,
            labels={'anio': 'Año', 'promedio': 'PM10 (µg/m³)'},
            template='plotly_dark',
            color_discrete_sequence=['#00CCCC'],
            title=f"Tendencia de PM10 en {departamento}"
        )
        st.plotly_chart(fig_pm10, use_container_width=True)
    else:
        st.info("No hay datos de PM10 para este departamento en el rango seleccionado.")

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

    if not pd.isna(promedio_pm10):
        categoria_pm10 = clasificar_pm10(promedio_pm10)
        if categoria_pm10 in ["Malo", "Muy Malo"]:
            st.error(f"El PM10 promedio ({promedio_pm10} µg/m³) indica **{categoria_pm10}** calidad del aire. Supera los límites recomendados y puede afectar la salud.")
        elif categoria_pm10 in ["Regular", "Aceptable"]:
            st.warning(f"El PM10 promedio ({promedio_pm10} µg/m³) indica **{categoria_pm10}** calidad del aire. Se recomienda monitoreo y control de emisiones.")
        else:
            st.success(f"El PM10 promedio ({promedio_pm10} µg/m³) indica **{categoria_pm10}** calidad del aire.")        

    if not pd.isna(promedio_irca):
        categoria_irca = clasificar_irca(promedio_irca)
        if categoria_irca in ["Riesgo alto", "Riesgo inviable sanitariamente"]:
            st.warning(f"El IRCA promedio ({promedio_irca}) indica **{categoria_irca}** en la calidad del agua.")
        elif categoria_irca in ["Riesgo medio", "Riesgo bajo"]:
            st.info(f"El IRCA promedio ({promedio_irca}) indica **{categoria_irca}** en la calidad del agua.")
        else:
            st.success(f"El IRCA promedio ({promedio_irca}) indica **{categoria_irca}** en la calidad del agua.")

    st.markdown("---")

    # ===============================================
    # Insights
    # ===============================================
    st.subheader("Insights")
    
    if not pd.isna(promedio_pm10):
        categoria_pm10 = clasificar_pm10(promedio_pm10)
        if categoria_pm10 == "Excelente":
            st.success("Los niveles de PM10 son excelentes y se encuentran por debajo de las recomendaciones de la OMS.")
        elif categoria_pm10 == "Bueno":
            st.success("La calidad del aire es buena en términos de PM10, aunque se recomienda mantener las acciones de control de emisiones.")
        elif categoria_pm10 == "Aceptable":
            st.info("La calidad del aire es aceptable. Es importante mantener vigilancia para evitar incrementos.")
        elif categoria_pm10 == "Regular":
            st.warning("La calidad del aire es regular según el promedio de PM10. Se encuentra dentro del límite legal, pero excede el valor recomendado por la OMS.")
        elif categoria_pm10 == "Malo":
            st.error("El nivel de PM10 supera los límites nacionales, indicando mala calidad del aire y posibles impactos en la salud.")
        elif categoria_pm10 == "Muy Malo":
            st.error("La calidad del aire es muy mala por niveles elevados de PM10. Se recomienda implementar medidas urgentes de reducción de emisiones.")

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