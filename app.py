# app.py
# ===============================================
# Streamlit App - Análisis de Calidad del Aire y Agua en Colombia
# Departamentos: [CÓRDOBA, CESAR, BOLÍVAR]
# ===============================================

import streamlit as st
#import pandas as pd
from src import loader, eda_aire, eda_agua, modelado

st.set_page_config(page_title="Calidad del Aire y Agua en Colombia", layout="wide")

st.title("Análisis de Calidad del Aire y Agua en Colombia")
st.write("""
Esta aplicación permite explorar interactivamente la calidad del aire y del agua en los departamentos de **Córdoba, Cesar y Bolívar** de Colombia, utilizando herramientas de análisis de datos en Python.
""")

# -------------------------------
# Sidebar - Menú de Navegación
# -------------------------------
menu = st.sidebar.selectbox(
    "Selecciona una sección",
    (
        "Carga y Limpieza de Datos", 
        "EDA - Calidad del Aire", 
        "EDA - Calidad del Agua", 
        "Modelado Predictivo", 
        "Análisis de Outliers PM10",
        "Análisis de Correlación PM10",
        "Modelado Avanzado",
        "Modelado Avanzado Completo",
        "Modelado mas Variables",
        "Modelado Avanzado Features"
    )
)

# -------------------------------
# Sección: Carga y limpieza
# -------------------------------
if menu == "Carga y Limpieza de Datos":
    st.header("Carga y limpieza de datos")
    df_aire, df_agua = loader.cargar_datos()

    st.subheader("Datos de Calidad del Aire (primeras filas)")
    st.dataframe(df_aire.head())

    st.subheader("Datos de Calidad del Agua (primeras filas)")
    st.dataframe(df_agua.head())

    st.info(f"Datos de aire: {df_aire.shape[0]} registros, {df_aire.shape[1]} columnas")
    st.info(f"Datos de agua: {df_agua.shape[0]} registros, {df_agua.shape[1]} columnas")

# -------------------------------
# Sección: EDA - Calidad del Aire
# -------------------------------
elif menu == "EDA - Calidad del Aire":
    st.header("Análisis Exploratorio de Datos - Calidad del Aire")
    df_aire, _ = loader.cargar_datos()
    eda_aire.eda_aire(df_aire, streamlit_mode=True)

# -------------------------------
# Sección: EDA - Calidad del Agua
# -------------------------------
elif menu == "EDA - Calidad del Agua":
    st.header("Análisis Exploratorio de Datos - Calidad del Agua")    
    _, df_agua = loader.cargar_datos()
    eda_agua.eda_agua(df_agua, streamlit_mode=True)
    

# -------------------------------
# Sección: Modelado Predictivo
# -------------------------------
elif menu == "Modelado Predictivo":
    st.header("Modelado Predictivo")

    st.write("""
    En esta sección se puede ejecutar un modelo de predicción de la calidad del aire
    utilizando regresión para proyectar valores de contaminantes a futuro.
    """)
    modelado.modelar_calidad_aire(streamlit_mode=True)
    modelado.modelar_calidad_agua(streamlit_mode=True)

elif menu == "Análisis de Outliers PM10":
    from src import analisis_outliers_pm10
    analisis_outliers_pm10.analisis_outliers_pm10()

elif menu == "Modelado Avanzado":
    from src import modelado_avanzado
    modelado_avanzado.modelar_avanzado()

elif menu == "Modelado Avanzado Completo":
    from src import modelado_avanzado_completo
    modelado_avanzado_completo.modelar_avanzado_completo()

elif menu == "Modelado mas Variables":
    from src import modelado_mas_variables
    modelado_mas_variables.ejecutar_modelado_avanzado_features()

elif menu == "Modelado Avanzado Features":
    from src import modelado_avanzado_features_streamlit_func
    modelado_avanzado_features_streamlit_func.modelado_avanzado_features_streamlit()


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("### Aplicación desarrollada por [César García, Luis Rodriguez y Rosalinda Parra] | Versión: 2025-06-28")