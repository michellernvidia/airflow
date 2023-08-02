import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

from ngc_requests import *
from task_workspace import *
from nemo_checkpoint import *
from pretrain_gpt import *
from download_squad import get_squad_dataset
from p_tuning import *

## 0. Variables
key_v = Variable.get("key_v", deserialize_json=True)
org_v = Variable.get("org_v", deserialize_json=True)
team_v = Variable.get("team_v", deserialize_json=True)
ace_v = Variable.get("ace_v", deserialize_json=True)
# workspace_name_v = Variable.get("workspace_name_v", deserialize_json=True)
nemo_ckpt_v = Variable.get("nemo_ckpt_v", deserialize_json=True)
pretrain_decision_v = Variable.get("pretrain_decision_v", deserialize_json=True)
tuning_method_v = Variable.get("tuning_method_v", deserialize_json=True)

key_= str(key_v)
org_=str(org_v)
team_= str(team_v)
ace_=str(ace_v)
# workspace_name_ = str(workspace_name_v)
nemo_ckpt_=str(nemo_ckpt_v)
pretrain_decision_ = str(pretrain_decision_v)
tuning_method_ = str(tuning_method_v)

gpt_workspace_name = "airflow_gpt_workspace"
tuning_workspace_name = f"airflow_{tuning_method_}_workspace"
    
## Define DAG + Tasks
with DAG(
         "LLM_WORKFLOW_NEMO_BCP", 
         schedule_interval='@once',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag: 

    create_gpt_workspace_task = PythonOperator(
            task_id = 'create_gpt_workspace',
            python_callable= create_gpt_workspace,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "workspace_name": gpt_workspace_name},
            dag = dag)
    
    create_tuning_workspace_task = PythonOperator(
            task_id = 'create_tuning_workspace',
            python_callable= create_tuning_workspace,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "workspace_name": tuning_workspace_name},
            dag = dag)
    
    pretrain_decision_task = BranchPythonOperator(
            task_id='get_base_model',
            provide_context=True,
            python_callable=get_base_model,
            op_kwargs={"pretrain_decision": pretrain_decision_},
            dag=dag)
    
    download_checkpoint_task = PythonOperator(
            task_id = 'download_nemo_checkpoint',
            python_callable= download_nemo_checkpoint,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_},
            dag = dag)

    # op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "workspace_name": workspace_name_, "team": team_},
    download_the_pile_task = PythonOperator(
            task_id = 'download_pile_dataset',
            python_callable= download_pile_dataset,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_},
            dag = dag)

    train_gpt_task = PythonOperator(
            task_id = 'train_gpt_model',
            python_callable= train_gpt_model,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_},
            dag = dag)
    
#     download_squad_task = PythonOperator(
#             task_id = 'download_squad_dataset',
#             python_callable=get_squad_dataset,
#             op_kwargs={"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_, "tuning_method": tuning_method_},
#             dag=dag
#     )

#     p_tuning_train_task = PythonOperator(
#             task_id = 'p_tuning_train',
#             python_callable= p_tuning_training_bcp,
#             op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_},
#             trigger_rule=TriggerRule.ONE_SUCCESS,
#             dag = dag)


create_gpt_workspace_task >> pretrain_decision_task
create_tuning_workspace_task >> pretrain_decision_task

pretrain_decision_task >> [download_checkpoint_task, download_the_pile_task]
download_the_pile_task >> train_gpt_task #>> p_tuning_train_task
download_checkpoint_task #>> p_tuning_train_task 

