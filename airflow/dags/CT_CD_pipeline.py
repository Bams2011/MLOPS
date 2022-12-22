# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from mlops_pipeline.data_prep import *
from mlops_pipeline.data_modeling import model_training,model_eval
#from mlops_pipeline.monitoring import *

fig_path = '/home/airflow/artifacts/'

# Declare Default arguments for the DAG
default_args = {
    'owner': 'bams@deloitte',
    'depends_on_past': False,
    'start_date': dt.datetime.strptime('2022-03-23T15:50:00', '%Y-%m-%dT%H:%M:%S'),
    'provide_context': True
}

# creating a new dag
CT_CD_dag = DAG('CT_CD_pipeline', default_args=default_args, schedule_interval='0 0 * * 2')

# Integrating read_data operator in airflow dag
load_data = PythonOperator(task_id='load_data', python_callable=download_data, dag=CT_CD_dag)
# Integrating data_preparation operator in airflow dag
prep_data = PythonOperator(task_id='prep_data', python_callable=data_prep, dag=CT_CD_dag)

# Integrating model_training operator in airflow dag
train_model = PythonOperator(task_id='train_model', python_callable=model_training, dag=CT_CD_dag)

# Integrating model_eval operator in airflow dag
eval_model = BranchPythonOperator(task_id='eval_model', python_callable=model_eval, dag=CT_CD_dag)

# Improved or regressed
improved = BashOperator( task_id="improved", bash_command="echo IMPROVED", dag=CT_CD_dag)
regressed = BashOperator(task_id="regressed",bash_command="echo REGRESSED",dag=CT_CD_dag)

# Serve model
commands = """
    echo served model;
    export MLFLOW_TRACKING_URI=http://localhost:5001;
    nohup mlflow models serve -m 'models:/fraud-detection/Staging' --no-conda --port 5002 --enable-mlserver > /home/airflow/output.log 2>&1 &
    sleep 10;
    exit
"""
deploy = BashOperator(
    task_id="deploy",  # push to GitHub to kick off serving workflows
    bash_command=commands,
    dag=CT_CD_dag
)

# Notifications (use appropriate operators, ex. EmailOperator)
report = BashOperator(task_id="report", bash_command="echo filed report",dag=CT_CD_dag)

# Task relationships
load_data >> prep_data >> train_model >> eval_model >> [improved, regressed]
improved >> deploy
regressed >> report


# monitoring tasks pipelines
"""monitoring_dag = DAG('monitoring_pipeline', default_args=default_args, schedule_interval='0 0 * * 2')

load_prod_data = PythonOperator(
		task_id="load_prod_data",
		python_callable=load_data_execute,
		provide_context=True,
        dag=monitoring_dag
	)

drift_analysis = PythonOperator(
    task_id="drift_analysis",
    python_callable=drift_analysis_execute,
    provide_context=True,
     dag=monitoring_dag
)

detect_drift = BranchPythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift_execute,
    provide_context=True,
    do_xcom_push=False, 
     dag=monitoring_dag
)

create_dashboard = PythonOperator(
    task_id='create_dashboard',
    provide_context=True,
    python_callable=create_dashboard_execute,
     dag=monitoring_dag
)

log_result = BashOperator(task_id="log_result",
 bash_command="echo No data drift detected",
 dag=monitoring_dag
 )

load_prod_data >> drift_analysis >> detect_drift >> [create_dashboard,log_result]"""