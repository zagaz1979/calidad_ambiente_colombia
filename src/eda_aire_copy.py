import matplotlib.pyplot as plt
import seaborn as sns

try:
    import streamlit as st
except ImportError:
    st = None

sns.set_theme(style="whitegrid")

def eda_aire(df_aire, streamlit_mode=False):
    if streamlit_mode and st:
        st.subheader("Vista general de los datos de calidad del aire")
        st.write(df_aire.head())
        st.info(f"Datos: {df_aire.shape[0]} registros, {df_aire.shape[1]} columnas")

    # Conversión de tipos
    if df_aire['anio'].dtype != 'int':
        df_aire['anio'] = df_aire['anio'].astype(int)

    contaminantes = [
        'PM10', 'O3', 'PST', 'P', 'PM2.5', 'TAire2',
        'SO2', 'NO2', 'CO', 'HAire2', 'DViento', 'RGlobal', 'VViento'
    ]

    for contaminante in contaminantes:
        if contaminante in df_aire['variable'].unique():
            plt.figure(figsize=(10, 5))
            sns.lineplot(
                data=df_aire[df_aire['variable'] == contaminante],
                x='anio',
                y='promedio',
                hue='departamento',
                marker='o'
            )
            plt.title(f'Evolución Anual de {contaminante}')
            plt.xlabel('Año')
            plt.ylabel(f'{contaminante} (µg/m³)')
            plt.legend(title='Departamento')
            plt.grid(True)
            plt.tight_layout()

            if streamlit_mode and st:
                st.pyplot(plt.gcf())
            else:
                plt.show()
        else:
            if not streamlit_mode:
                print(f"Contaminante '{contaminante}' no encontrado en el dataset.")

    # Pivot y barplot resumen
    df_pivot = df_aire.pivot_table(
        index=['departamento', 'anio'],
        columns='variable',
        values='promedio',
        aggfunc='mean'
    ).reset_index()

    df_promedios = df_pivot.drop(columns='anio').groupby('departamento').mean(numeric_only=True).reset_index()

    df_barplot = df_promedios.melt(
        id_vars='departamento',
        var_name='Contaminante',
        value_name='Valor promedio'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_barplot,
        x='Contaminante',
        y='Valor promedio',
        hue='departamento'
    )
    plt.title('Promedio Anual de Contaminantes por Departamento')
    plt.ylabel('µg/m³')
    plt.xlabel('Contaminante')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(plt.gcf())
    else:
        plt.show()

    # Análisis anual con catplot
    df_anual = (
        df_aire.groupby(['anio', 'variable', 'departamento'])['promedio']
        .mean()
        .reset_index()
        .rename(columns={'variable': 'Contaminante', 'promedio': 'Valor promedio'})
    )

    g = sns.catplot(
        data=df_anual,
        kind="bar",
        x="Contaminante",
        y="Valor promedio",
        hue="departamento",
        col="anio",
        col_wrap=2,
        height=5,
        aspect=1.5,
        palette="tab10"
    )
    g.set_titles("Año {col_name}")
    g.set_axis_labels("Contaminante", "Promedio (µg/m³)")
    g.set_xticklabels(rotation=45)
    plt.tight_layout()

    if streamlit_mode and st:
        st.pyplot(g.fig)
    else:
        plt.show()

    if not streamlit_mode:
        print("\nEDA de calidad del aire finalizado exitosamente.\n")

# Ejecución directa
if __name__ == "__main__":
    import loader_copy as loader_copy
    df_aire, _ = loader_copy.cargar_datos()
    eda_aire(df_aire)