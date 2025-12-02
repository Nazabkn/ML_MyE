from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/spaceflights"

KEDRO_UNDERSTANDING = f"cd {PROJECT_DIR} && kedro run --pipeline understanding"
KEDRO_PREPROCESSING = f"cd {PROJECT_DIR} && kedro run --pipeline preprocessing"
KEDRO_UNIFICATION = f"cd {PROJECT_DIR} && kedro run --pipeline unification"
KEDRO_FEATURE = f"cd {PROJECT_DIR} && kedro run --pipeline feature"
KEDRO_CLASSIF = f"cd {PROJECT_DIR} && kedro run --pipeline classification"
KEDRO_REGRESSION = f"cd {PROJECT_DIR} && kedro run --pipeline regression"
KEDRO_UNSUPERVISED = f"cd {PROJECT_DIR} && kedro run --pipeline unsupervised_learning"

with DAG(
    dag_id="spaceflights_kedro",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={"owner": "naza"},
    tags=["kedro", "ml"],
) as dag:
    understanding = BashOperator(
        task_id="run_understanding",
        bash_command=KEDRO_UNDERSTANDING,
    )

    preprocessing = BashOperator(
        task_id="run_preprocessing",
        bash_command=KEDRO_PREPROCESSING,
    )

    unification = BashOperator(
        task_id="run_unification",
        bash_command=KEDRO_UNIFICATION,
    )

    feature = BashOperator(
        task_id="run_feature",
        bash_command=KEDRO_FEATURE,
    )

    classification = BashOperator(
        task_id="run_classification",
        bash_command=KEDRO_CLASSIF,
    )

    evaluation = BashOperator(
        task_id="run_evaluation",
        bash_command=KEDRO_EVAL,
    )

    regression = BashOperator(
        task_id="run_regression",
        bash_command=KEDRO_REGRESSION,
    )

    unsupervised = BashOperator(
        task_id="run_unsupervised_learning",
        bash_command=KEDRO_UNSUPERVISED,
    )

    understanding >> preprocessing >> unification >> feature
    feature >> classification >> evaluation
    feature >> regression

    [evaluation, regression] >> unsupervised