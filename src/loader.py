"""
loader.py

Carga y filtrado de datos de calidad del aire y agua en Colombia,
renombrado de columnas, limpieza y normalización de nombres de
departamentos y municipios, filtrado de departamentos de interés
y guardado de CSV filtrados listos para análisis y modelado.

Autor: [César García, Luis Rodriguez y Rosalinda Parra]
Fecha: 2025-06-28
"""

import pandas as pd

# Configuración de rutas
ruta_aire = './data/calidad_aire_colombia.csv'
ruta_agua = './data/calidad_agua_colombia.csv'
ruta_guardado_aire = './data/aire_filtrado_caribe.csv'
ruta_guardado_agua = './data/agua_filtrada_caribe.csv'

# Departamentos de interés
departamentos_interes = ['CÓRDOBA', 'CESAR', 'BOLÍVAR']

def cargar_datos(streamlit_mode=False):
    print("Iniciando carga y filtrado de datos...\n")

    # Cargar datasets completos
    df_aire = pd.read_csv(ruta_aire)
    df_agua = pd.read_csv(ruta_agua)

    # Renombrar columnas de aire
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

    # Renombrar columnas de agua
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

    # Normalización de nombres
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

    # Normalizar texto
    for df in [df_aire, df_agua]:
        df['departamento'] = df['departamento'].str.strip().str.upper().replace(normalizacion_departamentos)
        df['municipio'] = df['municipio'].str.strip().str.upper()

    # Eliminar filas con '#TODOS'
    df_aire = df_aire[~df_aire.isin(['#TODOS']).any(axis=1)]
    df_agua = df_agua[~df_agua.isin(['#TODOS']).any(axis=1)]

    # Filtrado de departamentos de interés
    df_aire_filtrado = df_aire[df_aire['departamento'].isin(departamentos_interes)].copy()
    df_agua_filtrado = df_agua[df_agua['departamento'].isin(departamentos_interes)].copy()

    # Guardado CSVs filtrados
    df_aire_filtrado.to_csv(ruta_guardado_aire, index=False)
    df_agua_filtrado.to_csv(ruta_guardado_agua, index=False)

    print(f"Archivo filtrado de aire guardado en: {ruta_guardado_aire} ({df_aire_filtrado.shape[0]} registros)")
    print(f"Archivo filtrado de agua guardado en: {ruta_guardado_agua} ({df_agua_filtrado.shape[0]} registros)")
    print("\nCarga y filtrado completados correctamente.\n")

    # RETORNAR FILTRADOS
    return df_aire_filtrado, df_agua_filtrado

# Ejecución directa
if __name__ == "__main__":
    cargar_datos()