from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/spaceflights"

KEDRO_FEATURE = f"cd {PROJECT_DIR} && kedro run --pipeline feature"
KEDRO_CLASSIF = f"cd {PROJECT_DIR} && kedro run --pipeline classification"


with DAG(
    dag_id="spaceflights_kedro",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,      
    catchup=False,
    default_args={"owner": "naza"},
    tags=["kedro", "ml"],
) as dag:

    feature = BashOperator(
        task_id="run_feature",
        bash_command=KEDRO_FEATURE,
    )

    classification = BashOperator(
        task_id="run_classification",
        bash_command=KEDRO_CLASSIF,
    )


    feature >> classification
 
