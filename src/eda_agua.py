import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # Importación directa, ya que siempre estará disponible en Streamlit
import pandas as pd # Necesario para pd.isna y pd.api.types

sns.set_theme(style="whitegrid")

def clasificar_irca(valor):
    """
    Clasifica el valor del Índice de Riesgo de la Calidad del Agua (IRCA)
    en categorías de riesgo según la normativa.
    """
    if pd.isna(valor):
        return "Sin datos"
    if 0 <= valor <= 5:
        return 'Sin Riesgo'
    elif 5 < valor <= 14:
        return 'Riesgo Bajo'
    elif 14 < valor <= 35:
        return 'Riesgo Medio'
    elif 35 < valor <= 80:
        return 'Riesgo Alto'
    elif 80 < valor <= 100:
        return 'Riesgo Inviable Sanitariamente' # Corregido 'Invial' a 'Inviable Sanitariamente'
    else:
        return 'Valor fuera de rango'

def eda_agua(df_agua, streamlit_mode=False):
    """
    Realiza el Análisis Exploratorio de Datos (EDA) para la calidad del agua.
    Genera gráficos de distribución, evolución anual, clasificación de riesgo
    y top de municipios con peor calidad.

    Args:
        df_agua (pd.DataFrame): DataFrame con los datos de calidad del agua.
                                Se espera que la columna 'anio' ya sea de tipo datetime.
        streamlit_mode (bool): Si es True, los gráficos se muestran en Streamlit.
                               Si es False, se muestran con plt.show().
    """
    if streamlit_mode:
        st.subheader("Vista general de los datos de calidad del agua")
        st.write(df_agua.head())
        st.info(f"Datos: {df_agua.shape[0]} registros, {df_agua.shape[1]} columnas")
        # Asegurarse de que 'anio' es datetime, aunque loader.py debería manejarlo
        if not pd.api.types.is_datetime64_any_dtype(df_agua['anio']):
            st.warning("La columna 'anio' no es de tipo datetime. Se intentará convertir.")
            df_agua['anio'] = pd.to_datetime(df_agua['anio'], format='%Y', errors='coerce')
            df_agua.dropna(subset=['anio'], inplace=True) # Eliminar filas si la conversión falla


    # Verificar la existencia de la columna 'irca'
    if 'irca' not in df_agua.columns:
        error_msg = "Error: La columna 'irca' no se encuentra en el dataset de calidad del agua. No se puede realizar el EDA."
        if streamlit_mode:
            st.error(error_msg)
        else:
            print(error_msg)
        return # Salir de la función si la columna crítica no está

    # ===============================================
    # Boxplot: Distribución IRCA por año y departamento
    # ===============================================
    st.subheader("Distribución del IRCA por Año y Departamento")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_agua,
        x=df_agua['anio'].dt.year, # Usar el año numérico para el eje X
        y='irca',
        hue='departamento',
        palette='Set2',
        ax=ax
    )
    ax.set_title('Distribución del IRCA por Año y Departamento')
    ax.set_ylabel('IRCA (%)')
    ax.set_xlabel('Año')
    ax.legend(title='Departamento')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # Rotar etiquetas
    plt.tight_layout()

    if streamlit_mode:
        st.pyplot(fig)
    else:
        plt.show()
    plt.close(fig) # CERRAR LA FIGURA

    # ===============================================
    # Evolución del IRCA promedio por departamento
    # ===============================================
    st.subheader("Evolución del IRCA Promedio por Departamento")
    df_irca_anual = (
        df_agua.groupby([df_agua['anio'].dt.year, 'departamento'])['irca'] # Agrupar por año numérico
        .mean()
        .reset_index()
        .rename(columns={'anio': 'Año'}) # Renombrar la columna de año agrupada
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_irca_anual,
        x='Año', # Usar la columna 'Año' renombrada
        y='irca',
        hue='departamento',
        marker='o',
        ax=ax
    )
    ax.set_title('Evolución del IRCA Promedio por Departamento') # Título más genérico
    ax.set_ylabel('IRCA Promedio (%)')
    ax.set_xlabel('Año')
    ax.legend(title='Departamento')
    ax.set_xticks(df_irca_anual['Año'].unique()) # Asegurar que todos los años aparezcan como ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # Rotar etiquetas
    plt.tight_layout()

    if streamlit_mode:
        st.pyplot(fig)
    else:
        plt.show()
    plt.close(fig) # CERRAR LA FIGURA

    # ===============================================
    # Clasificación de riesgo según IRCA y conteo por clasificación
    # ===============================================
    st.subheader("Clasificación del IRCA por Departamento")
    
    # Crear una copia para evitar SettingWithCopyWarning al añadir la nueva columna
    df_agua_copy = df_agua.copy()
    df_agua_copy['CLASIFICACION_IRCA'] = df_agua_copy['irca'].apply(clasificar_irca)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df_agua_copy,
        x='CLASIFICACION_IRCA',
        order=['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto', 'Riesgo Inviable Sanitariamente', 'Sin datos', 'Valor fuera de rango'], # Orden completo
        hue='departamento',
        palette='Pastel1',
        ax=ax
    )
    ax.set_title('Clasificación del IRCA por Departamento')
    ax.set_ylabel('Número de Registros')
    ax.set_xlabel('Clasificación IRCA')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # Rotar etiquetas
    ax.legend(title='Departamento')
    plt.tight_layout()

    if streamlit_mode:
        st.pyplot(fig)
    else:
        plt.show()
    plt.close(fig) # CERRAR LA FIGURA

    # ===============================================
    # Top 10 municipios con peor calidad de agua
    # ===============================================
    st.subheader("Top 10 Municipios con Peor Calidad de Agua")
    df_municipios_irca = (
        df_agua.groupby('municipio')['irca']
        .mean()
        .reset_index()
        .sort_values(by='irca', ascending=False)
        .head(10)
    )

    if not df_municipios_irca.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_municipios_irca,
            x='irca',
            y='municipio',
            palette='Reds_r',
            ax=ax
        )
        ax.set_title('Top 10 Municipios con Peor Calidad de Agua (IRCA Promedio)')
        ax.set_xlabel('IRCA Promedio (%)')
        ax.set_ylabel('Municipio')
        plt.tight_layout()

        if streamlit_mode:
            st.pyplot(fig)
        else:
            plt.show()
        plt.close(fig) # CERRAR LA FIGURA
    else:
        if streamlit_mode:
            st.info("No hay datos de IRCA para mostrar el top de municipios en el rango seleccionado.")
        else:
            print("No hay datos de IRCA para mostrar el top de municipios en el rango seleccionado.")


    if not streamlit_mode:
        print("\nEDA de calidad del agua finalizado exitosamente.\n")

# Ejecución directa para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    # Importar loader correctamente desde src
    import src.loader as loader
    _, df_agua_test = loader.cargar_datos() # Cargar datos para la prueba
    if df_agua_test is not None:
        eda_agua(df_agua_test)
    else:
        print("No se pudieron cargar los datos para ejecutar el EDA de agua.")
