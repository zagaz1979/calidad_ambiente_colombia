# app.py
# ===============================================
# Streamlit App - Análisis de Calidad del Aire y Agua en Colombia
# Departamentos: [CÓRDOBA, CESAR, BOLÍVAR]
# ===============================================

import streamlit as st
from src import eda_aire, loader
from src import eda_agua # Importar solo los módulos necesarios inicialmente

# Configuración de la página de Streamlit
st.set_page_config(page_title="Calidad del Aire y Agua en Colombia", layout="wide")

st.title("Análisis de Calidad del Aire y Agua en Colombia")
st.write("""
Esta aplicación permite explorar interactivamente la calidad del aire y del agua en los departamentos de **Córdoba, Cesar y Bolívar** de Colombia, utilizando herramientas de análisis de datos en Python.
""")

# Cargar los datos una sola vez al inicio de la aplicación
# Streamlit cache_data asegura que los datos se carguen solo una vez
# a menos que los archivos subyacentes cambien.
@st.cache_data
def load_all_data():
    """Carga y preprocesa los datos de aire y agua."""
    df_aire, df_agua = loader.cargar_datos() # loader.cargar_datos() ahora debe retornar los DFs
    return df_aire, df_agua

df_aire, df_agua = load_all_data()

# -------------------------------
# Sidebar - Menú de Navegación
# -------------------------------
menu = st.sidebar.selectbox(
    "Selecciona una sección",
    (
        "EDA - Calidad del Aire", 
        "EDA - Calidad del Agua", 
        "Dashboard",
        "Modelado Predictivo (Simple)", 
        "Modelado Avanzado (Comparación)",
        "Modelado con Selección de Features"
    )
)

# -------------------------------
# Sección: EDA - Calidad del Aire
# -------------------------------
if menu == "EDA - Calidad del Aire":
    st.header("Análisis Exploratorio de Datos - Calidad del Aire")
    # eda_aire.eda_aire ahora debe aceptar df_aire como argumento
    eda_aire.eda_aire(df_aire, streamlit_mode=True)
    
# -------------------------------
# Sección: EDA - Calidad del Agua
# -------------------------------
elif menu == "EDA - Calidad del Agua":
    st.header("Análisis Exploratorio de Datos - Calidad del Agua")     
    # eda_agua.eda_agua ahora debe aceptar df_agua como argumento
    eda_agua.eda_agua(df_agua, streamlit_mode=True)

# -------------------------------
# Sección: Dashboard
# -------------------------------
elif menu == "Dashboard":
    # Importar el módulo del dashboard solo cuando sea necesario
    from src import dashboard 
    st.header("Dashboard General")
    st.write("Explora un resumen interactivo de la calidad del aire y del agua.")
    dashboard.mostrar_dashboard(df_aire, df_agua) # Pasar los DataFrames al dashboard
  
# -------------------------------
# Sección: Modelado Predictivo (Simple)
# -------------------------------
elif menu == "Modelado Predictivo (Simple)":
    st.header("Modelado Predictivo Simple")
    st.write("""
    Esta sección ejecuta un modelo de regresión lineal simple para predecir la evolución
    de contaminantes en el aire y el IRCA en el agua.
    """)
    # Importar el módulo de modelado solo cuando sea necesario
    from src import modelado 
    # modelado.modelar_calidad_aire y modelado.modelar_calidad_agua ahora deben aceptar DFs
    modelado.modelar_calidad_aire(df_aire, streamlit_mode=True)
    modelado.modelar_calidad_agua(df_agua, streamlit_mode=True)

# -------------------------------
# Sección: Modelado Avanzado (Comparación)
# -------------------------------
elif menu == "Modelado Avanzado (Comparación)":
    st.header("Modelado Avanzado - Comparación de Modelos")
    st.write("""
    Compara el rendimiento de Regresión Lineal, Regresión Polinómica y Random Forest
    para la predicción de PM10.
    """)
    # Importar el módulo de modelado avanzado completo solo cuando sea necesario
    from src import modelado_avanzado_completo
    # modelado_avanzado_completo.modelar_avanzado_completo ahora debe aceptar df_aire
    modelado_avanzado_completo.modelar_avanzado_completo(df_aire)

# -------------------------------
# Sección: Modelado con Selección de Features
# -------------------------------
elif menu == "Modelado con Selección de Features":
    st.header("Modelado con Selección de Features")
    st.write("""
    Permite seleccionar variables predictoras y comparar modelos (Lineal, Polinómica, Random Forest).
    """)
    # Importar el módulo de modelado avanzado con features solo cuando sea necesario
    from src import modelado_avanzado_features_streamlit_func
    # modelado_avanzado_features_streamlit_func.modelado_avanzado_features_streamlit ahora debe aceptar df_aire
    modelado_avanzado_features_streamlit_func.modelado_avanzado_features_streamlit(df_aire)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Aplicación desarrollada por [César García, Luis Rodriguez y Rosalina Parra] | Versión: 1.0.0")
