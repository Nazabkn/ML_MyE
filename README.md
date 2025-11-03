 ⭐ Proyecto ML Astronomía 🌙

Machine Learning con Kedro — Metodología CRISP-DM (Completo)

Este proyecto analiza datos astronómicos de asteroides y meteoritos para clasificar peligrosidad y predecir su diámetro promedio usando múltiples modelos de Machine Learning.

Incluye:
✅ Pipelines Kedro (Clasificación + Regresión + Reporting)
✅ Airflow para orquestación
✅ Docker para despliegue
✅ DVC + DagsHub para versionado de datos
✅ Modelado completo y evaluación 📊

｡𖦹°‧ Estructura del Proyecto ｡𖦹°‧

        spaceflights/
        │
        ├── conf/
        │   └── base/catalog.yml        # Definición de datasets versionados
        │
        ├── data/                       # CONTROLADO POR DVC ✅
        │   ├── 01_raw/
        │   ├── 02_intermediate/
        │   ├── 03_primary/
        │   ├── 05_model_input/
        │   ├── 06_models/
        │   ├── 07_model_output/
        │   └── 08_reporting/
        │
        ├── notebooks/
        │   ├── 01_business.ipynb
        │   ├── 02_data_understanding.ipynb
        │   ├── 03_preprocessing.ipynb
        │   └── 08_reporting/00_report.ipynb
        │
        ├── src/spaceflights/
        │   ├── pipelines/              # f01..f08 pipelines Kedro
        │   ├── daemon_airflow.py       # DAG de Airflow
        │   └── __init__.py
        │
        ├── Dockerfile
        ├── dvc.yaml
        ├── requirements.txt
        └── README.md


࣪ ִֶָ☾. Datasets utilizados ࣪ ִֶָ☾.

NEO
Fuente: NASA API
Descripción: Objetos cercanos a la Tierra

NEO_v2
Fuente: NASA 
Descripción: Velocidades / distancias

Meteorite Landings
Fuente: NASA Open Data 
Descripción: Registros reales de impacto

❀ CRISP-DM aplicado ❀

Fase 1 – Comprensión del negocio	
- 01_business.ipynb

Fase 2 – Comprensión de datos
- 02_data_understanding.ipynb

Fase 3 – Preparación de datos	
- 03_preprocessing.ipynb

Fase 4 – Modelado	
- Pipelines f05 y f07

Fase 5 – Evaluación		
- 08_reporting/00_report.ipynb

Fase 6 – Despliegue	
- Preparado para avanzarlo después


✦ Modelos implementados ✦

Clasificación — ¿Es peligroso el asteroide?

Modelos:

- Logistic Regression
- KNN
- Random Forest
- XGBoost/GradientBoost 
- SVC 

Regresión — ¿Predictor del diámetro del objeto?

Modelos:

- Linear Regression 
- Ridge 
- Lasso 
- Random Forest Regressor 
- SVR 

✮ Reportes & Gráficos ✮

- 08_reporting/ contiene:
  - Archivo	Contenido
  - confusion_matrix.png	Matriz de confusión final
  - roc_curve.png	Curva ROC
  - cv_bars.png	Comparación CV
  - reg_results_table.csv	Ranking de modelos de    regresión
  - final_classification_report.csv	Informe final sklearn


𔓘 Airflow 𔓘

Pipeline DAG ejecuta:
Clasificación → Regresión → Reportes

Ejecutar Airflow local:
airflow standalone


𓆡 Docker 𓆝

Construir imagen:
docker build -t kedro-spaceflights .

Ejecutar:
docker run -it kedro-spaceflights

☘︎ Versionado de Datos con DVC + DagsHub ☘︎

Descargar datos: 
dvc pull

Subir cambios:
dvc push

☀︎ Reproducibilidad completa ☀︎

git clone https://github.com/Nazabkn/ML_MyE.git
cd ML_MyE
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
kedro run
