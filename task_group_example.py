import datetime
from airflow import DAG
from airflow.decorators import task_group
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="dag1",
    start_date=datetime.datetime(2016, 1, 1),
    schedule="@daily",
    default_args={"retries": 1},
):

    @task_group()
    def group1():
        task1 = EmptyOperator(task_id="task1")
        task2 = EmptyOperator(task_id="task2")
        task3 = EmptyOperator(task_id="task3")

        task1 >> task3
        task2 >> task3


    task0 = EmptyOperator(task_id="task0")
    task4 = EmptyOperator(task_id="task4")

    task0 >> group1() >> task4