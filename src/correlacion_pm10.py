import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def correlacion_pm10():
    st.subheader("Análisis de Correlación de Contaminantes para PM10")

    df = pd.read_csv('./data/aire_filtrado_caribe.csv')

    # Filtrar Córdoba
    df = df[df['departamento'] == 'CÓRDOBA']

    # Pivot: contaminantes como columnas, promedio como valores
    df_pivot = df.pivot_table(
        index=['anio'],
        columns='variable',
        values='promedio',
        aggfunc='mean'
    ).reset_index()

    st.write("### Dataset de contaminantes pivotado:")
    st.dataframe(df_pivot.head())

    # Matriz de correlación
    df_corr = df_pivot.corr()

    st.write("### Matriz de Correlación")
    st.dataframe(df_corr)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Mapa de Calor de Correlación entre Contaminantes")
    st.pyplot(fig)

    st.success("Análisis de correlación completado. Utiliza las variables con mayor correlación positiva o negativa con PM10 como candidatos para modelado.")

# Ejecución local opcional
if __name__ == "__main__":
    correlacion_pm10()