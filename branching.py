from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import PythonOperator
# from airflow.operators.dummy import DummyOperator

from datetime import datetime
from airflow import DAG


def start_func(ti):
    print('Starting!!!!!!!!!')
    return


def branch_func(ti):
    xcom_value = 10
    if xcom_value >= 5:
        return 'continue_task'
    else:
        return 'stop_task'

def continue_func(ti):
    print('CONTINUING')
    return

def stop_func(ti):
    print('STOPPING')
    return

with DAG(
         "BRANCHING", 
         schedule_interval='@daily',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:

    
    start_op = PythonOperator(
            task_id = 'start_task',
            python_callable=start_func,
            dag = dag)

    branch_op = BranchPythonOperator(
                    task_id='branch_task',
                    provide_context=True,
                    python_callable=branch_func,
                    dag=dag)

    continue_op = PythonOperator(task_id='continue_task', 
                                 python_callable=continue_func,
                                 dag=dag)
    
    stop_op = PythonOperator(task_id='stop_task', 
                             python_callable=stop_func,
                             dag=dag)

start_op >> branch_op >> [continue_op, stop_op]