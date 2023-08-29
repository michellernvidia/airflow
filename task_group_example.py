from airflow.decorators import task_group
from airflow.operators import EmptyOperator


@task_group()
def group1():
    task1 = EmptyOperator(task_id="task1")
    task2 = EmptyOperator(task_id="task2")


task3 = EmptyOperator(task_id="task3")

group1() >> task3