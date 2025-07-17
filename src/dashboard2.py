import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from streamlit_extras.metric_cards import style_metric_cards
from . import loader_copy

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

def clasificar_ozono(valor):
    if pd.isna(valor):
        return "Sin datos"
    if valor <= 50:
        return "Bueno"
    elif valor <= 100:
        return "Moderado"
    elif valor <= 150:
        return "Dañino para sensibles"
    elif valor <= 200:
        return "Dañino"
    else:
        return "Muy dañino"

def mostrar_dashboard():
    st.title("Dashboard Calidad del Aire y Agua - Córdoba, Cesar y Bolívar")
    st.caption("Analiza IRCA, PM10, PM2.5, O3 y variables meteorológicas")
    
    df_aire, df_agua = loader_copy.cargar_datos(streamlit_mode=True)
    df_aire['anio'] = pd.to_datetime(df_aire['anio'], format='%Y')
    df_agua['anio'] = pd.to_datetime(df_agua['anio'], format='%Y')
    
    departamentos_interes = ['CÓRDOBA', 'CESAR', 'BOLÍVAR']
    departamento = st.selectbox("Selecciona el departamento:", departamentos_interes)
    min_anio = df_aire['anio'].dt.year.min()
    max_anio = df_aire['anio'].dt.year.max()
    rango_anios = st.slider("Selecciona rango de años:", min_anio, max_anio, (min_anio, max_anio), step=1)
    
    df_aire_filtrado = df_aire[(df_aire['departamento'] == departamento) &
                                (df_aire['anio'].dt.year >= rango_anios[0]) &
                                (df_aire['anio'].dt.year <= rango_anios[1])]
    df_agua_filtrado = df_agua[(df_agua['departamento'] == departamento) &
                                (df_agua['anio'].dt.year >= rango_anios[0]) &
                                (df_agua['anio'].dt.year <= rango_anios[1])]

    st.subheader("KPIs por Promedios")
    col1, col2, col3, col4 = st.columns(4)

    promedio_pm10 = df_aire_filtrado[df_aire_filtrado['variable'] == 'PM10']['promedio'].mean()
    promedio_pm25 = df_aire_filtrado[df_aire_filtrado['variable'] == 'PM2.5']['promedio'].mean()
    promedio_o3 = df_aire_filtrado[df_aire_filtrado['variable'] == 'O3']['promedio'].mean()
    promedio_irca = df_agua_filtrado['irca'].mean()
    
    col1.metric("PM10 Promedio", f"{promedio_pm10:.2f}" if not pd.isna(promedio_pm10) else "Sin datos")
    col2.metric("PM2.5 Promedio", f"{promedio_pm25:.2f}" if not pd.isna(promedio_pm25) else "Sin datos")
    col3.metric("O3 Promedio", f"{promedio_o3:.2f}" if not pd.isna(promedio_o3) else "Sin datos")
    col4.metric("IRCA Promedio", f"{promedio_irca:.2f}" if not pd.isna(promedio_irca) else "Sin datos")
    
    style_metric_cards(background_color="#111", border_left_color="#007070", border_color="#AAAAAA")

    st.subheader("Tendencias Temporales")
    variable = st.selectbox("Selecciona variable para analizar tendencia:", ['PM10', 'PM2.5', 'O3', 'IRCA'])
    if variable == 'IRCA':
        if not df_agua_filtrado.empty:
            df_grouped = df_agua_filtrado.groupby(df_agua_filtrado['anio'].dt.year)['irca'].mean().reset_index()
            fig = px.line(df_grouped, x='anio', y='irca', markers=True,
                          title=f"Tendencia del IRCA en {departamento}",
                          labels={'anio':'Año', 'irca':'IRCA'},
                          template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    else:
        df_variable = df_aire_filtrado[df_aire_filtrado['variable'] == variable]
        if not df_variable.empty:
            df_grouped = df_variable.groupby(df_variable['anio'].dt.year)['promedio'].mean().reset_index()
            fig = px.line(df_grouped, x='anio', y='promedio', markers=True,
                          title=f"Tendencia de {variable} en {departamento}",
                          labels={'anio':'Año', 'promedio':f'{variable} Promedio'},
                          template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa Coroplético por Municipio (IRCA)")
    if not df_agua_filtrado.empty:
        df_mapa = df_agua_filtrado.groupby(['municipio'])['irca'].mean().reset_index()
        fig_mapa = px.choropleth(df_mapa, locations='municipio', locationmode='geojson-id',
                                 color='irca', color_continuous_scale='Turbo',
                                 title=f"Mapa de IRCA promedio por municipio en {departamento}")
        st.plotly_chart(fig_mapa, use_container_width=True)

    st.subheader("Datos Filtrados")
    st.dataframe(df_aire_filtrado, use_container_width=True)
    st.dataframe(df_agua_filtrado, use_container_width=True)
