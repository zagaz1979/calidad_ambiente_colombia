# Proyecto: Evolución de la Calidad del Aire y del Agua en Córdoba, Cesar y Bolívar

## Descripción General

Este proyecto de análisis de datos aborda el desafío ambiental de comprender y predecir la evolución de la calidad del aire y del agua en los departamentos de Córdoba, Cesar y Bolívar en Colombia. Utilizando datos históricos de fuentes oficiales, hemos desarrollado una herramienta interactiva y modelos predictivos para identificar tendencias, factores influyentes y áreas críticas, con el fin de apoyar la toma de decisiones informadas para la gestión ambiental y la salud pública.

## Integrantes del Equipo

* César García
* Rosalina Parra
* Luis Miguel Rodríguez Zabala

**Facultad:** Ingeniería
**Programa Académico:** Análisis de Datos
**Universidad:** Universidad de Bolívar

## Contexto y Problema

Los departamentos presentan retos ambientales relacionados con contaminación de fuentes
hídricas y calidad del aire, generando impactos en la salud pública. Sin un análisis de datos
estructurado, resulta difícil diseñar políticas públicas efectivas y estrategias de mitigación
basadas en evidencia.

## Pregunta de Investigación

¿Cómo ha evolucionado la calidad del aire y del agua en Córdoba, Cesar y Bolívar en los últimos años y cuáles son los factores que más influyen en su variación?

## Objetivos

### Objetivo General
Evaluar y comprender la evolución temporal de la calidad del aire y del agua en los departamentos de Córdoba, Cesar y Bolívar, identificando tendencias, causas y posibles impactos ambientales y de salud.

### Objetivos Específicos
1.  Identificar las problemáticas actuales de calidad de aire y agua en Córdoba, Cesar y Bolívar.
2.  Desarrollar un sistema digital de integración de datos para el monitoreo continuo.
3.  Implementar visualizaciones gráficas para facilitar el análisis y la toma de decisiones.

## Metodología

El proyecto se estructuró en fases interconectadas para garantizar un análisis de datos completo y riguroso:

1.  **Obtención y Preparación de Datos:**
    * Utilización de datasets públicos y oficiales de calidad del aire y agua de entidades como IDEAM, SIAC y Datos.gov.co.
    **Aire**
    * https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Calidad-Del-Aire-En-Colombia-Promedio-Anual-/kekd-7v7h/about_data
    **Agua**
    * https://www.datos.gov.co/Salud-y-Protecci-n-Social/Calidad-del-Agua-para-Consumo-Humano-en-Colombia/nxt2-39c3/about_data
    * Carga, filtrado (para los departamentos de interés: Córdoba, Cesar, Bolívar) y limpieza de datos, incluyendo la estandarización de nombres, manejo de valores faltantes y tratamiento de valores atípicos (outliers).

2.  **Análisis Exploratorio de Datos (EDA):**
    * Cálculo de estadísticas descriptivas para todas las variables relevantes.
    * Análisis de distribuciones anuales y tendencias de contaminantes (PM10, PM2.5, O3, NO2, SO2, CO, PLiquida, entre otros) e indicadores de calidad del agua (IRCA).
    * Generación de boxplots por departamento y año para visualizar distribuciones y diferencias regionales/temporales.

3.  **Modelado Predictivo:**
    * Aplicación de modelos de aprendizaje automático para comprender y predecir la evolución de la calidad del aire y del agua.
    * **Modelos utilizados:** Regresión Lineal (como línea base), Regresión Polinómica (para relaciones no lineales) y Random Forest (robusto para relaciones complejas y múltiples variables).
    * **Evaluación:** Los modelos se entrenaron y evaluaron utilizando métricas como R-cuadrado (R²), Error Cuadrático Medio (RMSE) y Error Absoluto Medio (MAE).

## Herramientas y Tecnologías

* **Python:** Lenguaje de programación principal para el análisis de datos.
    * **Pandas:** Manipulación y análisis de datos (DataFrames).
    * **NumPy:** Soporte para operaciones numéricas.
    * **Matplotlib & Seaborn:** Creación de gráficos estáticos para EDA.
    * **Plotly:** Generación de gráficos interactivos para dashboards.
    * **Scikit-learn:** Implementación de algoritmos de aprendizaje automático (Regresión Lineal, Polinómica, Random Forest).
* **Streamlit:** Framework de Python para construir la aplicación web interactiva y los dashboards.
* **Power BI:** Herramienta de Business Intelligence para la creación de dashboards ejecutivos y reportes detallados.
* **IDE:** Visual Studio Code
* **Control de Versiones:** GitHub

## Hallazgos Clave

### Calidad del Aire
* **Tendencias de Contaminantes:** Se observaron fluctuaciones significativas en contaminantes como PM10 y PM2.5 a lo largo de los años en los departamentos de estudio. Por ejemplo, Bolívar ha presentado picos más altos de PM10 en ciertos años, indicando episodios de mayor contaminación.
* **Factores Influyentes:** Las variaciones anuales sugieren la influencia de factores como la estacionalidad (épocas secas vs. lluvias) y el aumento de la actividad vehicular e industrial.

### Calidad del Agua
* **Evolución del IRCA:** El Índice de Riesgo de la Calidad del Agua (IRCA) promedio mostró una evolución diversa entre 2007 y 2023, con algunos departamentos evidenciando mejoras o deterioros.
* **Municipios con Mayor Riesgo:** El análisis permite identificar los municipios con los índices de riesgo más elevados, crucial para priorizar intervenciones.

### Rendimiento de los Modelos
* El modelo **Random Forest** generalmente mostró un mejor ajuste a los valores reales en comparación con los modelos Lineal y Polinómico, lo que sugiere que las relaciones en los datos de calidad del aire y agua son complejas y no lineales.

## Evaluación y Análisis del Proyecto

### Estudio de Factibilidad
El proyecto es altamente factible. Se apoya en la disponibilidad de datos públicos, el uso de tecnologías robustas de código abierto (Python, Streamlit) y se alinea con el marco regulatorio ambiental colombiano. La experiencia del equipo y la interfaz intuitiva de la herramienta garantizan su viabilidad técnica y operativa.

### Estudio de Análisis de Riesgos
Se identificaron riesgos relacionados con la calidad y disponibilidad de datos (mitigado con limpieza y monitoreo), riesgos técnicos (mitigado con pruebas y optimización de código) y riesgos operacionales (mitigado con capacitación y promoción).

### Evaluación Multidimensional
* **Financiera:** Bajos costos directos de desarrollo gracias a herramientas de código abierto. Beneficios indirectos en optimización de recursos y prevención de costos en salud.
* **Ambiental:** Impacto directo y positivo al facilitar la gestión y comprensión de la contaminación, promoviendo políticas sostenibles y monitoreo continuo.
* **Socioeconómica:** Mejora la salud pública, fomenta la conciencia ciudadana y apoya el desarrollo económico sostenible al proporcionar información para decisiones informadas.

## Conclusiones y Recomendaciones

### Conclusiones
* Se identificaron departamentos con mayor deterioro en la calidad del aire y del agua.
* Se establecieron las variables críticas de calidad del aire (PM10, PM2.5, O3, NO2, SO2, CO) y de agua (IRCA) para un monitoreo continuo.
* Los hallazgos y visualizaciones contribuyen a una toma de decisiones ambientales más informada.

### Recomendaciones
1.  Incrementar puntos de monitoreo en zonas críticas para una recolección de datos más densa y en tiempo real.
2.  Promover y fortalecer políticas de reducción de emisiones y protección de fuentes hídricas.
3.  Integrar los modelos predictivos en sistemas de alerta temprana para anticipar periodos de alta contaminación y tomar medidas preventivas en salud pública.

## Estructura del Proyecto (Archivos)

El proyecto está organizado en la siguiente estructura de directorios y archivos principales:


.
├── src/
│   ├── loader.py             # Carga y preprocesamiento inicial de los datos.
│   ├── dashboard.py          # Lógica para el dashboard principal de Streamlit.
│   ├── eda_aire.py           # Funciones para el Análisis Exploratorio de Datos del aire.
│   ├── eda_agua.py           # Funciones para el Análisis Exploratorio de Datos del agua.
│   ├── modelado.py           # Funciones para el modelado de regresión lineal simple.
│   ├── modelado_avanzado_completo.py # Funciones para modelado avanzado (Polinómica, Random Forest).
│   └── modelado_avanzado_features_streamlit_func.py # Funciones adicionales de modelado avanzado para Streamlit.
├── app.py                    # Archivo principal de la aplicación Streamlit.
├── README.md                 # Este archivo.
├── requirements.txt          # Dependencias del proyecto.
├── Proyecto calidad aire y agua.pdf # Documento completo del proyecto.
├── 1-Documentación columnas calidad del aire en Colombia.pdf # Documentación de columnas (aire).
├── 2-Documentación columnas calidad del agua.pdf # Documentación de columnas (agua).
└── data/                     # Directorio para los datasets (no incluidos en el repo, se asume carga externa).


## Cómo Ejecutar la Aplicación Streamlit

Para ejecutar la aplicación Streamlit de este proyecto localmente, sigue los siguientes pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/zagaz1979/calidad-ambiente-colombia.git](https://github.com/zagaz1979/calidad-ambiente-colombia.git)
    cd calidad-ambiente-colombia
    ```

2.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate   # En Windows
    ```

3.  **Instalar las Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con todas las librerías necesarias. Si no lo tienes, puedes crearlo con:
    ```bash
    pip freeze > requirements.txt
    ```
    Luego, instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    (Asegúrate de que `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `streamlit-extras` estén en tu `requirements.txt`).

4.  **Asegurar los Datos:**
    Coloca tus archivos `calidad_aire_colombia.csv` y `calidad_agua_colombia.csv` dentro de un directorio llamado `data/` en la raíz del proyecto, o asegúrate de que tu `loader.py` apunte a la ubicación correcta de tus datos.

5.  **Ejecutar la Aplicación Streamlit:**
    ```bash
    streamlit run app.py
    ```

    Esto abrirá la aplicación en tu navegador web predeterminado.

## Contacto

Para cualquier pregunta o comentario sobre este proyecto, no dudes en contactar a los integrantes del equipo.
