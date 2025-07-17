import pandas as pd
import os # Importar para verificar la existencia de archivos y crear directorios
import streamlit as st # Importar para mostrar mensajes en Streamlit si es necesario

# Configuración de rutas
# Rutas de los archivos CSV originales
ruta_aire = './data/calidad_aire_colombia.csv'
ruta_agua = './data/calidad_agua_colombia.csv'

# Rutas para guardar los datasets filtrados (opcional, para persistencia)
ruta_guardado_aire = './data/aire_filtrado_caribe.csv'
ruta_guardado_agua = './data/agua_filtrada_caribe.csv'

# Departamentos de interés para el filtrado
departamentos_interes = ['CÓRDOBA', 'CESAR', 'BOLÍVAR']

def cargar_datos(streamlit_mode=False):
    """
    Carga, preprocesa y filtra los datasets de calidad del aire y agua.
    Realiza renombrado de columnas, normalización de texto, manejo de valores
    especiales y conversión de tipos de datos.

    Args:
        streamlit_mode (bool): Si es True, muestra mensajes de error/información
                               usando Streamlit. Si es False, usa print.

    Returns:
        tuple: Una tupla que contiene (df_aire_filtrado, df_agua_filtrado).
               Retorna (None, None) si ocurre un error crítico al cargar los datos.
    """
    # Mensaje inicial de carga
    if streamlit_mode:
        st.info("Iniciando carga y preprocesamiento de datos...")
    else:
        print("Iniciando carga y preprocesamiento de datos...\n")

    # ===============================================
    # Cargar datasets completos con manejo de errores
    # ===============================================
    try:
        df_aire = pd.read_csv(ruta_aire)
        df_agua = pd.read_csv(ruta_agua)
    except FileNotFoundError as e:
        error_msg = f"Error: No se encontró el archivo de datos. Asegúrate de que los archivos '{ruta_aire}' y '{ruta_agua}' existan. Detalle: {e}"
        if streamlit_mode:
            st.error(error_msg)
        else:
            print(error_msg)
        return None, None # Retorna None para indicar que la carga falló
    except pd.errors.EmptyDataError as e:
        error_msg = f"Error: Uno de los archivos de datos está vacío. Detalle: {e}"
        if streamlit_mode:
            st.error(error_msg)
        else:
            print(error_msg)
        return None, None
    except Exception as e:
        error_msg = f"Ocurrió un error inesperado al cargar los datos. Detalle: {e}"
        if streamlit_mode:
            st.error(error_msg)
        else:
            print(error_msg)
        return None, None

    # ===============================================
    # Renombrar columnas de aire
    # ===============================================
    df_aire.rename(columns={
        'ID Estacion': 'id_estacion',
        'Autoridad Ambiental': 'autoridad_ambiental',
        'Estación': 'estacion',
        'Latitud': 'latitud',
        'Longitud': 'longitud',
        'Variable': 'variable',
        'Unidades': 'unidades',
        'Tiempo de exposición (horas)': 'tiempo_exposicion',
        'Año': 'anio',
        'Promedio': 'promedio',
        'Suma': 'suma',
        'No. de datos': 'n_datos',
        'Representatividad Temporal': 'representatividad',
        'Excedencias limite actual': 'excedencias',
        'Porcentaje excedencias limite actual': 'porcentaje_excedencias',
        'Mediana': 'mediana',
        'Percentil 98': 'percentil_98',
        'Máximo': 'maximo',
        'Fechas/horas del máximo': 'fecha_maximo',
        'Mínimo': 'minimo',
        'Fechas/horas del mínimo': 'fecha_minimo',
        'Días de excedencias': 'dias_excedencias',
        'Código del Departamento': 'cod_departamento',
        'Nombre del Departamento': 'departamento',
        'Código del Municipio': 'cod_municipio',
        'Nombre del Municipio': 'municipio',
        'Tipo de Estación': 'tipo_estacion',
        'Ubicacion': 'ubicacion'
    }, inplace=True)

    # ===============================================
    # Renombrar columnas de agua
    # ===============================================
    df_agua.rename(columns={
        'DepartamentoCodigo': 'cod_departamento',
        'Departamento': 'departamento',
        'MunicipioCodigo': 'cod_municipio',
        'Municipio': 'municipio',
        'Año': 'anio',
        'IRCA': 'irca',
        'Nivel de riesgo': 'nivel_riesgo',
        'IRCAurbano': 'irca_urbano',
        'Nivel de riesgo urbano': 'nivel_riesgo_urbano',
        'IRCArural': 'irca_rural',
        'Nivel de riesgo rural': 'nivel_riesgo_rural'
    }, inplace=True)

    # ===============================================
    # Normalización de nombres de departamentos y municipios
    # ===============================================
    normalizacion_departamentos = {
        'BOYACA': 'BOYACÁ',
        'QUINDIO': 'QUINDÍO',
        'ATLANTICO': 'ATLÁNTICO',
        'CORDOBA': 'CÓRDOBA',
        'BOLIVAR': 'BOLÍVAR',
        'BOGOTA, D.C.': 'BOGOTÁ, D.C.',
        'CAQUETA': 'CAQUETÁ',
        'CHOCO': 'CHOCÓ',
    }

    for df in [df_aire, df_agua]:
        # Eliminar espacios en blanco y convertir a mayúsculas
        df['departamento'] = df['departamento'].str.strip().str.upper().replace(normalizacion_departamentos)
        df['municipio'] = df['municipio'].str.strip().str.upper()

    # ===============================================
    # Eliminar filas con '#TODOS' (valores no deseados)
    # ===============================================
    df_aire = df_aire[~df_aire.isin(['#TODOS']).any(axis=1)].copy() # Usar .copy() para evitar SettingWithCopyWarning
    df_agua = df_agua[~df_agua.isin(['#TODOS']).any(axis=1)].copy() # Usar .copy() para evitar SettingWithCopyWarning

    # ===============================================
    # Conversión de la columna 'anio' a tipo datetime (CRÍTICO)
    # Esto asegura que todos los módulos reciban 'anio' en el formato correcto.
    # ===============================================
    df_aire['anio'] = pd.to_datetime(df_aire['anio'], format='%Y', errors='coerce')
    df_agua['anio'] = pd.to_datetime(df_agua['anio'], format='%Y', errors='coerce')

    # Eliminar filas donde la conversión de 'anio' falló (si errors='coerce' se usó)
    df_aire.dropna(subset=['anio'], inplace=True)
    df_agua.dropna(subset=['anio'], inplace=True)

    # ===============================================
    # Filtrado de departamentos de interés
    # ===============================================
    df_aire_filtrado = df_aire[df_aire['departamento'].isin(departamentos_interes)].copy()
    df_agua_filtrado = df_agua[df_agua['departamento'].isin(departamentos_interes)].copy()

    # ===============================================
    # Guardado de CSVs filtrados (Opcional: para persistencia/depuración)
    # = Considera si realmente necesitas guardar estos archivos cada vez.
    # Si la aplicación solo usa los DataFrames en memoria, puedes omitir esto.
    # ===============================================
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True) # Asegura que el directorio 'data' exista

    try:
        df_aire_filtrado.to_csv(ruta_guardado_aire, index=False)
        df_agua_filtrado.to_csv(ruta_guardado_agua, index=False)
        if streamlit_mode:
            st.success(f"Archivos filtrados guardados en: {ruta_guardado_aire} y {ruta_guardado_agua}")
        else:
            print(f"Archivo filtrado de aire guardado en: {ruta_guardado_aire} ({df_aire_filtrado.shape[0]} registros)")
            print(f"Archivo filtrado de agua guardado en: {ruta_guardado_agua} ({df_agua_filtrado.shape[0]} registros)")
    except Exception as e:
        error_msg = f"Advertencia: No se pudieron guardar los archivos filtrados. Detalle: {e}"
        if streamlit_mode:
            st.warning(error_msg)
        else:
            print(error_msg)

    if streamlit_mode:
        st.success("Carga y preprocesamiento completados correctamente.")
    else:
        print("\nCarga y preprocesamiento completados correctamente.\n")

    # Retornar los DataFrames filtrados y preprocesados
    return df_aire_filtrado, df_agua_filtrado

# Ejecución directa para pruebas locales (fuera de Streamlit)
if __name__ == "__main__":
    df_aire_test, df_agua_test = cargar_datos()
    if df_aire_test is not None and df_agua_test is not None:
        print("\nPrimeras 5 filas del DataFrame de aire filtrado:")
        print(df_aire_test.head())
        print("\nInformación del DataFrame de aire filtrado:")
        df_aire_test.info()
        print("\nPrimeras 5 filas del DataFrame de agua filtrada:")
        print(df_agua_test.head())
        print("\nInformación del DataFrame de agua filtrada:")
        df_agua_test.info()