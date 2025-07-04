import matplotlib.pyplot as plt
import seaborn as sns

try:
    import streamlit as st
except ImportError:
    st = None

sns.set_theme(style="whitegrid")

def clasificar_irca(valor):
    if valor <= 5:
        return 'Sin Riesgo'
    elif valor <= 14:
        return 'Riesgo Bajo'
    elif valor <= 35:
        return 'Riesgo Medio'
    elif valor <= 80:
        return 'Riesgo Alto'
    else:
        return 'Riesgo Invial'

def eda_agua(df_agua, streamlit_mode=False):
    if streamlit_mode and st:
        st.subheader("Vista general de los datos de calidad del agua")
        st.write(df_agua.head())
        st.info(f"Datos: {df_agua.shape[0]} registros, {df_agua.shape[1]} columnas")

    if 'irca' not in df_agua.columns:
        raise ValueError("La columna 'irca' no se encuentra en el dataset.")

    # Boxplot: Distribución IRCA por año y departamento
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_agua,
        x='anio',
        y='irca',
        hue='departamento',
        palette='Set2'
    )
    plt.title('Distribución del IRCA por Año y Departamento')
    plt.ylabel('IRCA (%)')
    plt.xlabel('Año')
    plt.legend(title='Departamento')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    # Evolución del IRCA promedio por departamento
    df_irca_anual = (
        df_agua.groupby(['anio', 'departamento'])['irca']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_irca_anual,
        x='anio',
        y='irca',
        hue='departamento',
        marker='o'
    )
    plt.title('Evolución del IRCA Promedio por Departamento (2007–2023)')
    plt.ylabel('IRCA Promedio (%)')
    plt.xlabel('Año')
    plt.legend(title='Departamento')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    # Clasificación de riesgo según IRCA
    df_agua['CLASIFICACION_IRCA'] = df_agua['irca'].apply(clasificar_irca)

    # Conteo por clasificación
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df_agua,
        x='CLASIFICACION_IRCA',
        order=['Sin Riesgo', 'Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto', 'Riesgo Invial'],
        hue='departamento',
        palette='Pastel1'
    )
    plt.title('Clasificación del IRCA por Departamento')
    plt.ylabel('Número de Registros')
    plt.xlabel('Clasificación IRCA')
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    # Top 10 municipios con peor calidad de agua
    df_municipios_irca = (
        df_agua.groupby('municipio')['irca']
        .mean()
        .reset_index()
        .sort_values(by='irca', ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_municipios_irca,
        x='irca',
        y='municipio',
        palette='Reds_r'
    )
    plt.title('Top 10 Municipios con Peor Calidad de Agua (IRCA Promedio)')
    plt.xlabel('IRCA Promedio (%)')
    plt.ylabel('Municipio')
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    if not streamlit_mode:
        print("\nEDA de calidad del agua finalizado exitosamente.\n")

# Ejecución directa
if __name__ == "__main__":
    import src.loader as loader
    _, df_agua = loader.cargar_datos()
    eda_agua(df_agua)