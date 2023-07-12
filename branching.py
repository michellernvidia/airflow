from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

import datetime
from airflow import DAG

def branch_func(**kwargs):
    ti = kwargs['ti']
    xcom_value = int(ti.xcom_pull(task_ids='start_task'))
    if xcom_value >= 5:
        return 'continue_task'
    else:
        return 'stop_task'


with DAG(
         "BRANCHING", 
         schedule_interval='@daily',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:

    start_op = BashOperator(
                    task_id='start_task',
                    bash_command="echo 5",
                    xcom_push=True,
                    dag=dag)

    branch_op = BranchPythonOperator(
                    task_id='branch_task',
                    provide_context=True,
                    python_callable=branch_func,
                    dag=dag)

    continue_op = DummyOperator(task_id='continue_task', dag=dag)
    stop_op = DummyOperator(task_id='stop_task', dag=dag)

start_op >> branch_op >> [continue_op, stop_op]