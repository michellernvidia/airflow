from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.empty import EmptyOperator
# from airflow.decorators import dag, task
# from airflow.utils.edgemodifier import Label
from airflow.operators.python import BranchPythonOperator


def get_base_model():
    return

def choose_fine_tuning_method():
    return
    
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
    
    download_anthropic_hh_task = EmptyOperator(
            task_id = 'download_anthropic_hh_dataset',
            dag = dag)

    train_gpt_task = EmptyOperator(
            task_id = 'train_gpt_model',
            dag = dag)

    tuning_decision_task = BranchPythonOperator(
            task_id='tune_base_model',
            provide_context=True,
            python_callable=choose_fine_tuning_method,
            dag=dag)
    
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
    
    inference_task = EmptyOperator(
            task_id = 'inference_on_squad',
            dag = dag)
    
    rlhf_inference_task = EmptyOperator(
            task_id = 'supported_rlhf_inference',
            dag = dag)

pretrain_decision_task >> [download_checkpoint_task, download_the_pile_task]
download_the_pile_task >> train_gpt_task >> tuning_decision_task
download_checkpoint_task >> tuning_decision_task

tuning_decision_task >> download_squad_task >> p_tuning_train_task >> inference_task
tuning_decision_task >> download_squad_task >> sft_train_task >> inference_task
tuning_decision_task >> download_squad_task >> lora_train_task >> inference_task

tuning_decision_task >> download_squad_task >> sft_train_task >> p_tuning_train_task >> inference_task
tuning_decision_task >> download_squad_task >> sft_train_task >> download_anthropic_hh_task >> rlhf_rm_train_task >> rlhf_ppo_task >> rlhf_inference_task


