from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task
from airflow.utils.edgemodifier import Label


start_main_dag = EmptyOperator(task_id='start_main_dag', dag=dag)
    
## Define DAG + Tasks
with DAG(
         "model_alignment_flowchart", 
         schedule_interval='@once',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag: 

    pretrain_decision_task = BranchPythonOperator(
            task_id='get_base_model',
            provide_context=True,
            python_callable=get_base_model,
            op_kwargs={"pretrain_decision": pretrain_decision_},
            dag=dag)
        
    download_checkpoint_task = EmptyOperator(
            task_id = 'download_nemo_checkpoint',
            dag = dag)

    download_the_pile_task = EmptyOperator(
            task_id = 'download_pile_dataset',
            dag = dag)
    
    download_squad_task = EmptyOperator(
            task_id = 'download_squad_dataset',
            dag = dag)

    train_gpt_task = EmptyOperator(
            task_id = 'train_gpt_model',
            dag = dag)

    p_tuning_train_task = EmptyOperator(
            task_id = 'p_tuning_train',
            dag = dag)
    
    sft_train_task = EmptyOperator(
            task_id = 'sft_train',
            dag = dag)
    
    lora_train_task = EmptyOperator(
            task_id = 'lora_train',
            dag = dag)
    
    rlhf_rm_train_task = EmptyOperator(
            task_id = 'rlhf_rm_train',
            dag = dag)
    
    rlhf_ppo_task = EmptyOperator(
            task_id = 'rlhf_ppo',
            dag = dag)

pretrain_decision_task >> [download_checkpoint_task, download_the_pile_task]
download_the_pile_task >> train_gpt_task >> p_tuning_train_task
download_checkpoint_task >> p_tuning_train_task 

