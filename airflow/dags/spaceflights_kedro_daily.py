from __future__ import annotations
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/spaceflights"


CMD_FEATURE = f"cd {PROJECT_DIR} && kedro run --pipeline feature"
CMD_CLASSIF = f"cd {PROJECT_DIR} && kedro run --pipeline classification"
CMD_EVAL = f"cd {PROJECT_DIR} && kedro run --pipeline evaluation"
CMD_REGRESSION = f"cd {PROJECT_DIR} && kedro run --pipeline regression"
CMD_UNDERSTANDING = f"cd {PROJECT_DIR} && kedro run --pipeline understanding"
CMD_PREPROCESSING = f"cd {PROJECT_DIR} && kedro run --pipeline preprocessing"
CMD_UNIFICATION = f"cd {PROJECT_DIR} && kedro run --pipeline unification"

with DAG(
    dag_id="spaceflights_kedro_daily",
    description="Orquesta el flujo completo de Kedro para clasificación y regresión",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 9 * * *",  
    catchup=False,
    default_args={
        "owner": "naza",
        "retries": 0,
    },
    tags=["kedro", "ml", "daily"],
) as dag:
        understanding = BashOperator(
        task_id="kedro_understanding",
        bash_command=CMD_UNDERSTANDING,
    )

    preprocessing = BashOperator(
        task_id="kedro_preprocessing",
        bash_command=CMD_PREPROCESSING,
    )

    unification = BashOperator(
        task_id="kedro_unification",
        bash_command=CMD_UNIFICATION,
    )

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
    


    understanding >> preprocessing >> unification >> feature
    feature >> classification >> evaluation
    feature >> regression