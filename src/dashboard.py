import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from streamlit_extras.metric_cards import style_metric_cards
from src import loader

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

    if not pd.isna(promedio_pm10) and promedio_pm10 > 50:
        st.error(f"PM10 promedio ({promedio_pm10} µg/m³) excede el límite recomendado por OMS (50 µg/m³).")
    if not pd.isna(promedio_irca) and promedio_irca > 40:
        st.warning(f"El IRCA promedio ({promedio_irca}) indica riesgo en la calidad del agua.")

    st.markdown("---")

    # ===============================================
    # Insights
    # ===============================================
    st.subheader("Insights")
    if not pd.isna(promedio_pm10) and promedio_pm10 > 50:
        st.info("Los niveles elevados de PM10 sugieren evaluar fuentes de emisión en este departamento.")
    if not pd.isna(promedio_irca) and promedio_irca > 40:
        st.info("El IRCA elevado puede reflejar falta de tratamiento o contaminación en fuentes de agua local.")