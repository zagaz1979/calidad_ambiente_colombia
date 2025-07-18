import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd 

sns.set_theme(style="whitegrid")

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

def eda_aire(df_aire, streamlit_mode=False):
    """
    Realiza el Análisis Exploratorio de Datos (EDA) para la calidad del aire.
    Genera gráficos de evolución anual, promedios por departamento y análisis anual.

    Args:
        df_aire (pd.DataFrame): DataFrame con los datos de calidad del aire.
                                Se espera que la columna 'anio' ya sea de tipo datetime.
        streamlit_mode (bool): Si es True, los gráficos se muestran en Streamlit.
                               Si es False, se muestran con plt.show().
    """
    if streamlit_mode:
        st.subheader("Vista general de los datos de calidad del aire")
        st.write(df_aire.head())
        st.info(f"Datos: {df_aire.shape[0]} registros, {df_aire.shape[1]} columnas")
        # Asegurarse de que 'anio' es datetime para operaciones de fecha
        # Aunque loader.py debería manejarlo, esta es una verificación de respaldo.
        if not pd.api.types.is_datetime64_any_dtype(df_aire['anio']):
            st.warning("La columna 'anio' no es de tipo datetime. Se intentará convertir.")
            df_aire['anio'] = pd.to_datetime(df_aire['anio'], format='%Y', errors='coerce')
            df_aire.dropna(subset=['anio'], inplace=True) # Eliminar filas si la conversión falla

    # --- CAMBIO CLAVE AQUÍ: Obtener contaminantes dinámicamente ---
    # Obtener todas las variables únicas que tienen datos no nulos en 'promedio'
    # y que también están en nuestro diccionario de unidades.
    # Esto asegura que solo se intenten graficar variables con datos válidos y unidades conocidas.
    contaminantes_para_plotear = df_aire.dropna(subset=['promedio'])['variable'].unique().tolist()
    # Filtrar para incluir solo las que tienen una unidad definida en UNIDADES_CONTAMINANTES
    contaminantes_para_plotear = [c for c in contaminantes_para_plotear if c in UNIDADES_CONTAMINANTES]


    st.subheader("Evolución Anual de Contaminantes por Departamento")
    # Iterar sobre la lista dinámica de contaminantes
    for contaminante in contaminantes_para_plotear:
        # Filtrar el DataFrame para el contaminante actual
        # Usar .copy() para evitar SettingWithCopyWarning
        df_contaminante = df_aire[df_aire['variable'] == contaminante].copy()

        if not df_contaminante.empty:
            fig, ax = plt.subplots(figsize=(10, 5)) # Crear figura y ejes
            sns.lineplot(
                data=df_contaminante,
                x=df_contaminante['anio'].dt.year, # Usar el año numérico para el eje X
                y='promedio',
                hue='departamento',
                marker='o',
                ax=ax # Pasar los ejes al plot
            )
            ax.set_title(f'Evolución Anual de {contaminante}')
            ax.set_xlabel('Año')
            
            # Obtener la unidad correcta del diccionario
            unidad = UNIDADES_CONTAMINANTES.get(contaminante, 'Unidad Desconocida')
            ax.set_ylabel(f'{contaminante} Promedio ({unidad})')
            
            ax.legend(title='Departamento')
            ax.grid(True)
            plt.tight_layout()

            if streamlit_mode:
                st.pyplot(fig)
            else:
                plt.show()
            plt.close(fig) # CERRAR LA FIGURA para liberar memoria
        else:
            if streamlit_mode:
                st.info(f"Contaminante '{contaminante}' no encontrado en el dataset para el análisis de evolución.")
            # No se necesita un 'else' para print si streamlit_mode es False, ya que no se imprime nada.

    # ===============================================
    # Promedio General de Contaminantes por Departamento (Barplot)
    # ===============================================
    st.subheader("Promedio General de Contaminantes por Departamento")
    
    # Calcular promedio de 'promedio' por departamento y variable
    df_promedios_vars = df_aire.groupby(['departamento', 'variable'])['promedio'].mean().reset_index()

    # Filtrar solo los contaminantes que están en la lista UNIDADES_CONTAMINANTES
    # Esto ya incluye PLiquida si está en el diccionario y en los datos.
    df_promedios_vars_filtrado = df_promedios_vars[df_promedios_vars['variable'].isin(UNIDADES_CONTAMINANTES.keys())]

    if not df_promedios_vars_filtrado.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_promedios_vars_filtrado,
            x='variable',
            y='promedio',
            hue='departamento',
            ax=ax
        )
        ax.set_title('Promedio General de Contaminantes por Departamento')
        ax.set_ylabel('Valor Promedio') # Etiqueta genérica ya que las unidades varían
        ax.set_xlabel('Contaminante')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # Rotar etiquetas para mejor lectura
        ax.legend(title='Departamento')
        plt.tight_layout()

        if streamlit_mode:
            st.pyplot(fig)
        else:
            plt.show()
        plt.close(fig) # CERRAR LA FIGURA
    else:
        if streamlit_mode:
            st.info("No hay datos de promedios de contaminantes para mostrar el gráfico general.")

    # ===============================================
    # Análisis anual con catplot (promedio por año y contaminante)
    # ===============================================
    st.subheader("Promedio Anual de Contaminantes por Año y Departamento")

    # Agrupar por año, variable y departamento para obtener el promedio
    df_anual_grouped = (
        df_aire.groupby([df_aire['anio'].dt.year, 'variable', 'departamento'])['promedio']
        .mean()
        .reset_index()
        .rename(columns={'anio': 'Año', 'variable': 'Contaminante', 'promedio': 'Valor promedio'})
    )

    # Filtrar solo los contaminantes que están en la lista UNIDADES_CONTAMINANTES
    # Esto ya incluye PLiquida si está en el diccionario y en los datos.
    df_anual_grouped_filtrado = df_anual_grouped[df_anual_grouped['Contaminante'].isin(UNIDADES_CONTAMINANTES.keys())]

    if not df_anual_grouped_filtrado.empty:
        # Crear el catplot
        g = sns.catplot(
            data=df_anual_grouped_filtrado,
            kind="bar",
            x="Contaminante",
            y="Valor promedio",
            hue="departamento",
            col="Año", # Ahora usa la columna 'Año' que es numérica
            col_wrap=2,
            height=5,
            aspect=1.5,
            palette="tab10",
            sharey=False # Importante si las unidades varían mucho entre contaminantes
        )
        g.set_titles("Año {col_name}")
        
        # Iterar sobre los ejes para establecer etiquetas de Y dinámicas si es posible,
        # o mantener una etiqueta genérica si hay muchas unidades diferentes en el mismo panel.
        # Por simplicidad, mantendremos una etiqueta genérica para el eje Y,
        # y el título del gráfico ya indica el contexto.
        g.set_axis_labels("Contaminante", "Valor Promedio")
        g.set_xticklabels(rotation=45, ha='right')
        plt.tight_layout()

        if streamlit_mode:
            st.pyplot(g.fig)
        else:
            plt.show()
        plt.close(g.fig) # CERRAR LA FIGURA
    else:
        if streamlit_mode:
            st.info("No hay datos anuales de contaminantes para mostrar el catplot.")

    if not streamlit_mode:
        print("\nEDA de calidad del aire finalizado exitosamente.\n")

# Ejecución directa para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    # Importar loader correctamente desde src
    import src.loader as loader
    df_aire_test, _ = loader.cargar_datos() # Cargar datos para la prueba
    if df_aire_test is not None:
        eda_aire(df_aire_test, streamlit_mode=True) # Ejecutar en modo Streamlit para ver el comportamiento
    else:
        print("No se pudieron cargar los datos para ejecutar el EDA.")
