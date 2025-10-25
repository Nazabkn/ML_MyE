from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/spaceflights"


CMD_FEATURE = f"cd {PROJECT_DIR} && kedro run --pipeline feature"
CMD_CLASSIF = f"cd {PROJECT_DIR} && kedro run --pipeline classification"
CMD_EVAL = f"cd {PROJECT_DIR} && kedro run --pipeline evaluation"
CMD_REGRESSION = f"cd {PROJECT_DIR} && kedro run --pipeline regression"


with DAG(
    dag_id="spaceflights_kedro_daily",
    description="Corre pipelines de Feature, Classification y Evaluation de Kedro diariamente",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 9 * * *",  
    catchup=False,
    default_args={
        "owner": "naza",
        "retries": 0,
    },
    tags=["kedro", "ml", "daily"],
) as dag:


    feature = BashOperator(
        task_id="kedro_feature",
        bash_command=CMD_FEATURE,
    )

  
    classification = BashOperator(
        task_id="kedro_classification",
        bash_command=CMD_CLASSIF,
    )

    
    evaluation = BashOperator(
        task_id="kedro_evaluation",
        bash_command=CMD_EVAL,
    )

   
    regression = BashOperator(
        task_id="kedro_regression",
        bash_command=CMD_REGRESSION,
    )
    

    
    feature >> classification >> evaluation
    feature >> regression
