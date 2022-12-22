from atexit import register
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account
from fraud_detection.main import register_model
from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.python_operator import PythonOperator,BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from config import config
import pandas as pd
from sklearn.model_selection import train_test_split
from fraud_detection import main

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}

GE_ROOT_DIR = Path(config.BASE_DIR, "tests", "great_expectations")


def _extract_from_dwh():
    """Extract data from data_dir 
    (it will act as our BigQuery data warehouse) and
    save it locally.
    """
    
    # Extract data
    df = pd.read_csv(Path(config.RAW_DATA))
    #Transform data
    train_data,test_data =  train_test_split(df, df[config.CLASS_NAME], test_size = 0.2, random_state = 42)
    #load data
    train_data.to_csv(config.TRAIN_DATA)
    test_data.to_csv(config.TEST_DATA)


@dag(
    dag_id="mlops",
    description="implement MLOps level 1 architecture tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["mlops"],
)
def mlops():
    """MLOps workflows."""
    extract_from_dwh = PythonOperator(
        task_id="extract_data",
        python_callable=_extract_from_dwh,
    )
    validate_data = GreatExpectationsOperator(
        task_id="validate",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )
    # optimize = PythonOperator(
    #     task_id="optimize",
    #     python_callable=main.optimize,
    #     op_kwargs={
    #         "args_fp": Path(config.CONFIG_DIR, "args.json"),
    #         "study_name": "optimization",
    #         "num_trials": 1,
    #     },
    # )
    train = PythonOperator(
        task_id="train",
        python_callable=main.train_model,
        op_kwargs={
            "args_fp": Path(config.CONFIG_DIR, "args.json"),
            "experiment_name": "baselines",
            "run_name": "rf",
        },
    )
    evaluate_model = BranchPythonOperator(
        task_id='evaluate_model', 
        python_callable=main.evaluate_model
    )
    improved = BashOperator(
        task_id="improved", 
        bash_command="echo IMPROVED"
    )
    regressed = BashOperator(
        task_id="regressed",
        bash_command="echo REGRESSED"
    )
    report = BashOperator(
        task_id="report",
        bash_command="echo report from last training"
    )
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=main.register_model
    )

    # Define DAG
    extract_from_dwh >> validate_data >> train >> evaluate_model >> [improved, regressed]
    improved >> register_model
    regressed >> report
    
# Run DAGs
ml = mlops()