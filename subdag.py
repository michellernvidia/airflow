from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.subdag import SubDagOperator

def subdag(parent_dag_name, child_dag_name, args):
    # Define the SubDAG
    dag_subdag = DAG(
        dag_id=f'{parent_dag_name}.{child_dag_name}',
        default_args=args,
        schedule_interval="@daily",  # Set the schedule_interval for the SubDAG
    )

    # Define tasks for the SubDAG
    with dag_subdag:
        start_task = EmptyOperator(task_id='start_task')
        
        def print_hello():
            print("Hello from the SubDAG!")

        hello_task = PythonOperator(
            task_id='hello_task',
            python_callable=print_hello
        )

        end_task = EmptyOperator(task_id='end_task')

        # Define the task dependencies
        start_task >> hello_task >> end_task

    return dag_subdag

# Define the main DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'main_dag_with_subdag',
    default_args=default_args,
    schedule_interval="@daily",
)

# Define tasks for the main DAG
start_main_dag = EmptyOperator(task_id='start_main_dag', dag=dag)

# Create the SubDAG using the SubDagOperator
subdag_task = SubDagOperator(
    task_id='subdag_task',
    subdag=subdag('main_dag_with_subdag', 'subdag_task', default_args),
    dag=dag,
)

end_main_dag = EmptyOperator(task_id='end_main_dag', dag=dag)

# Define the task dependencies for the main DAG
start_main_dag >> subdag_task >> end_main_dag
