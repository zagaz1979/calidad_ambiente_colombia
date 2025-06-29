"""
main.py

Pipeline principal del proyecto de Análisis de Calidad de Aire y Agua en Colombia.
Ejecuta:
    - Limpieza avanzada
    - EDA aire
    - EDA agua
    - Modelado aire
    - Modelado agua

Autor: [César García, Luis Rodriguez y Rosalinda Parra]
Fecha: 2025-06-28
"""

import os
import time

from src import limpieza_avanzada
from src import eda_aire
from src import eda_agua
from src import modelado

# ==========================================
# UTILIDAD: Limpieza de consola cross-OS
# ==========================================
def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================
# FUNCIONES DE EJECUCIÓN ORDENADA
# ==========================================
def ejecutar_pipeline_completo():
    limpiar_consola()
    print("Iniciando Pipeline de Análisis de Calidad de Aire y Agua en Colombia\n")
    time.sleep(1)

    # Paso 1: Limpieza avanzada
    print("Paso 1: Ejecutando limpieza avanzada de datos...\n")
    limpieza_avanzada.limpieza_avanzada()
    time.sleep(1)

    # Paso 2: Análisis exploratorio de calidad del aire
    print("\nPaso 2: Ejecutando EDA de calidad del aire...\n")
    eda_aire.eda_aire()
    time.sleep(1)

    # Paso 3: Análisis exploratorio de calidad del agua
    print("\nPaso 3: Ejecutando EDA de calidad del agua...\n")
    eda_agua.eda_agua()
    time.sleep(1)

    # Paso 4: Modelado de calidad del aire
    print("\nPaso 4: Ejecutando modelado de calidad del aire...\n")
    modelado.modelar_calidad_aire()
    time.sleep(1)

    # Paso 5: Modelado de calidad del agua
    print("\nPaso 5: Ejecutando modelado de calidad del agua...\n")
    modelado.modelar_calidad_agua()
    time.sleep(1)

    print("\nPipeline ejecutado exitosamente. Todos los análisis y gráficos generados.\n")

# ==========================================
# EJECUCIÓN DIRECTA
# ==========================================
if __name__ == "__main__":
    ejecutar_pipeline_completo()