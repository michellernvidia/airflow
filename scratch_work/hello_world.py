from datetime import datetime
from airflow import DAG
# from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import PythonOperator

def print_hello():
    print('Hello world from first Airflow DAG!')

with DAG('hello_world', 
    description='Hello World DAG',
    schedule_interval='0 12 * * *', 
    start_date=datetime(2017, 3, 20), 
    catchup=False) as dag:
    
    hello_operator = PythonOperator(task_id='hello_task', 
    python_callable=print_hello, dag=dag)

hello_operator