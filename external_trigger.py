from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

def my_task_function():
    # Your task logic goes here
    print("External Triggered Task Executed!")

def my_task_function():
    # Your task logic goes here
    print("Checking blah blah blah..........")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG with external triggers
dag = DAG(
    'external_trigger_example',
    default_args=default_args,
    schedule_interval=None,  # Disable the default schedule interval
)

# Define the tasks in the DAG
start_task = EmptyOperator(task_id='start_task', dag=dag)

external_trigger_task = PythonOperator(
    task_id='external_trigger_task',
    python_callable=my_task_function,
    dag=dag,
)

external_trigger_task_2 = PythonOperator(
    task_id='external_trigger_task',
    python_callable=my_task_function,
    dag=dag,
)

end_task = EmptyOperator(task_id='end_task', dag=dag)

# Define the task dependencies
start_task >> external_trigger_task >> end_task

external_trigger_task_2
