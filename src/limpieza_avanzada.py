"""
limpieza_avanzada.py

Este módulo realiza:
Renombrado adicional y normalización de columnas.
Normalización avanzada de municipios con correcciones específicas.
Exploración de valores únicos en cada columna.
Eliminación de inconsistencias comunes.
Generación de reportes de limpieza para auditar los datasets.

Autor: [César García, Luis Rodriguez y Rosalinda Parra]
Fecha: 2025-06-28
"""

import pandas as pd

# Configuración de rutas de los CSV ya filtrados
ruta_aire = './data/aire_filtrado_caribe.csv'
ruta_agua = './data/agua_filtrada_caribe.csv'

ruta_guardado_aire = './data/aire_limpio.csv'
ruta_guardado_agua = './data/agua_limpia.csv'

def normalizar_municipios(df, columna='municipio', correcciones=None):
    """
    Normaliza los nombres de municipios en un DataFrame.
    """
    df[columna] = df[columna].str.strip().str.upper()
    if correcciones:
        df[columna] = df[columna].replace(correcciones)
    return df

def limpieza_avanzada():
    print("Iniciando limpieza avanzada de datos...\n")

    # Cargar datasets ya filtrados
    df_aire = pd.read_csv(ruta_aire)
    df_agua = pd.read_csv(ruta_agua)

    print("Valores únicos de departamentos en aire:")
    print(sorted(df_aire['departamento'].unique()))
    print("\nValores únicos de departamentos en agua:")
    print(sorted(df_agua['departamento'].unique()))

    # Correcciones específicas para municipios en aire
    correcciones_municipios_aire = {
        'AGUSTIN CODAZZI': 'AGUSTÍN CODAZZI',
        'AMAGA': 'AMAGÁ',
        'CHIRIGUANA': 'CHIRIGUANÁ',
        'ITAGUI': 'ITAGÜÍ',
        'JARDIN': 'JARDÍN',
        'IBAGUE': 'IBAGUÉ',
        'MEDELLIN': 'MEDELLÍN',
        'MONTELIBANO': 'MONTELÍBANO',
        'MONTERIA': 'MONTERÍA',
        'NEMOCON': 'NEMOCÓN',
        'POPAYAN': 'POPAYÁN',
        'CARTAGENA DE INDIAS': 'CARTAGENA',
        'PUERTO BERRIO': 'PUERTO BERRÍO',
        'SAN JOSE DE URE': 'SAN JOSÉ DE URÉ',
        'SAN JOSE DE LA MONTAÑA': 'SAN JOSÉ DE LA MONTAÑA',
        'SIBATE': 'SIBATÉ',
        'SONSON': 'SONSÓN',
        'SOPETRAN': 'SOPETRÁN',
        'SOPO': 'SOPÓ',
        'TOCANCIPA': 'TOCANCIPÁ',
        'TULUA': 'TULUÁ',
        'YONDO': 'YONDÓ',
        'ZIPAQUIRA': 'ZIPAQUIRÁ'
    }
    df_aire = normalizar_municipios(df_aire, correcciones=correcciones_municipios_aire)

    # Correcciones específicas para municipios en agua (si se detectan)
    correcciones_municipios_agua = {
        # Completa con valores detectados en auditoría si es necesario
    }
    df_agua = normalizar_municipios(df_agua, correcciones=correcciones_municipios_agua)

    # Ver valores únicos de municipios tras limpieza
    print("\nMunicipios únicos en aire tras limpieza:")
    print(sorted(df_aire['municipio'].dropna().unique()))

    print("\nMunicipios únicos en agua tras limpieza:")
    print(sorted(df_agua['municipio'].dropna().unique()))

    # Reporte de valores únicos en cada columna del dataset aire
    print("\nValores únicos por columna en df_aire:")
    for col in df_aire.columns:
        print(f"\nColumna: {col}")
        print(df_aire[col].unique())

    # Reporte de valores únicos en cada columna del dataset agua
    print("\nValores únicos por columna en df_agua:")
    for col in df_agua.columns:
        print(f"\nColumna: {col}")
        print(df_agua[col].unique())

    # Guardar datasets limpios
    df_aire.to_csv(ruta_guardado_aire, index=False)
    df_agua.to_csv(ruta_guardado_agua, index=False)

    print(f"\nDataset de aire limpio guardado en: {ruta_guardado_aire} ({df_aire.shape[0]} registros)")
    print(f"Dataset de agua limpio guardado en: {ruta_guardado_agua} ({df_agua.shape[0]} registros)")
    print("\nLimpieza avanzada completada correctamente.\n")

# Ejecución directa opcional para pruebas
if __name__ == "__main__":
    limpieza_avanzada()