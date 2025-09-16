 ⭐ Proyecto ML Astronomía – CRISP-DM ⭐

Este repositorio contiene un proyecto de Machine Learning desarrollado con Kedro, siguiendo la metodología CRISP-DM hasta la Fase 3 (Preparación de los Datos). El enfoque está en la exploración y preparación de datos astronómicos (asteroides y meteoritos) para futuros modelos predictivos.
___________________________________________________________________________

♡  ∩____∩ 
  („• ֊ •„)♡
|￣U U￣￣￣￣￣￣￣￣￣|
|  Estructura!        |   
￣￣￣￣￣￣￣￣￣￣￣￣

spaceflights/
  conf/                  (Configuración de Kedro)
      base/catalog.yml    (Catálogo de datasets)

   data/
     01_raw/              (Datos originales)
       02_intermediate/   (Datos limpios y preprocesados)
       03_primary/        (Datos unificados)

   notebooks/             Notebooks de análisis
       01_business.ipynb
       02_data_understanding.ipynb
       03_preprocessing.ipynb

   src/spaceflights/      
        pipelines/
        data_processing/
        __ini__.py

 requirements.txt       
 README.md             
 dvc.yaml / .dvc/       

___________________________________________________________________________


｡ﾟﾟ･ ｡ ･ﾟﾟ ｡ 
ﾟ。Datasets utilizados:
　ﾟ･｡ ･ﾟ 

1. Near Earth Objects (NEO) API de NASA


2. Neo_v2 (Venia en conjunto con NEO)


3. Meteorite Landings



Estos datasets se encuentran en data/01_raw luego fueron procesados para que esten en el data/03_primary :D


___________________________________________________________________________


𐙚¡El uso de la metodología CRISP-DM en el proyecto!𐙚
 
Fase 1 – Comprensión del Negocio

La definición

Los objetivos

La documentación del plan en notebooks/01_business.ipynb :D
--------------------------------------------------------------------

Fase 2 – Comprensión de los Datos

Recolección de 3 datasets astronómicos

Exploración inicial de datos con estadísticas y visualizaciones

Habian dos dataset igual asi :'v 

Identificación de valores nulos, tipos de variables y calidad de los datos
----------------------------------------------------------------------

Fase 3 – Preparación de los Datos

Limpieza de columnas con valores nulos

´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
Extracción de año desde el nombre de los asteroides

Promedio del diámetro estimado

Transformaciones logarítmicas
´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´

Unificación de meteoritos y asteroides por año en model_input_table.parquet

Resultados documentados en 03_preprocessing.ipynb



___________________________________________________________________________


 /\ _/\
( • . •)
/~     \  Instalación y ejecución

1.- Clonar el repositorio

git clone https://github.com/Nazabkn/ML_MyE.git
cd ML_MyE

2.- Crear entorno virtual e instalar dependencias

python -m venv .venv
.venv\Scripts\activate      (Windows)

lo siento profe no hay para mac :P!, 
broma, si hay: 

source .venv/bin/activate    (Mac o linux)

pip install -r requirements.txt

3. Ejecutar Kedro

Para correr todo el pipeline:

kedro run


4. Sincronizar datos con DVC

Este repo usa DVC con DagsHub para almacenar datos versionados:

dvc pull   esta es para descargar datos
dvc push   y esta para subir cambios

___________________________________________________________________________


  ✿   Documentación    ✿

Los notebooks sirven como bitácora de trabajo y prototipado
(No use casi ningún comentario, estaba fascinada con los comandos que encontraba en internet)

La ejecución automatizada está en los pipelines de Kedro


___________________________________________________________________________


---Estado del proyecto---   (˶╹ᵕ╹˶) kirby
 
Este proyecto se entrega hasta Fase 3 de CRISP-DM
Las fases posteriores (Modelado, Evaluación y Despliegue) las haré pronto, saludos!