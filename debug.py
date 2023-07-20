import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

from ngc_requests import *

## 0. Variables
key_v = Variable.get("key_v", deserialize_json=True)
org_v = Variable.get("org_v", deserialize_json=True)
team_v = Variable.get("team_v", deserialize_json=True)
ace_v = Variable.get("ace_v", deserialize_json=True)
workspace_name_v = Variable.get("workspace_name_v", deserialize_json=True)
nemo_ckpt_v = Variable.get("nemo_ckpt_v", deserialize_json=True)
pretrain_decision_v = Variable.get("pretrain_decision_v", deserialize_json=True)

key_= str(key_v)
org_=str(org_v)
team_= str(team_v)
ace_=str(ace_v)
workspace_name_ = str(workspace_name_v)
nemo_ckpt_=str(nemo_ckpt_v)
pretrain_decision_ = str(pretrain_decision_v)

def p_tuning_training_bcp(ti, org, ace, team=None):
      
      #get workspace id
      workspace_id = 'BZQLetfySFuBjdIHee6fLg' #anotha-airflow-wksp
      
      #ngc job parameters
      job_name = "p_tune_airflow_debug"
      ace_instance = "dgxa100.80g.4.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspace_mount_path = "/mount/workspace"
      job_command = "python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
                            prompt_learning=gpt3/squad \
                            stages=[prompt_learning] \
                            cluster_type=bcp \
                            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
                            data_dir=/mount/workspace \
                            base_results_dir=/mount/workspace/results \
                            prompt_learning.run.model_train_name=gpt3_5b \
                            prompt_learning.trainer.devices=4 \
                            prompt_learning.model.language_model_path=/mount/workspace/gpt_models/nemo_gpt5B_bf16_tp2.nemo \
                            prompt_learning.model.tensor_model_parallel_size=2 \
                            >> /results/prompt_learning_gpt3_log.txt 2>&1"
      
      #send ngc job request
      job_response = ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspace_mount_path, workspace_id, job_command, team=team)
      job_id = job_response['job']['id']

      return job_response, job_id


def wait_for_job_completion(ti, org, team=None):

     _, job_id = ti.xcom_pull(task_ids='p_tuning_train')
     job_status = ngc_job_status(ti, org, job_id)

     min=0
     while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED' and job_status != 'KILLED_BY_USER':
         time.sleep(300) #increase wait time to 5 mins
         min += 5
         job_status = ngc_job_status(ti, org, job_id)
         print(f'minute: {min} | Job status: ', job_status)
     return job_status


## Define DAG + Tasks
with DAG(
         "P_TUNING_DEBUG_JOB_STATUS", 
         schedule_interval='@once',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag:

    token_task = PythonOperator(
            task_id = 'token',
            python_callable=get_token,
            op_kwargs={"key": key_, "org": org_ , "team": team_},
            dag = dag
    ) 

    p_tuning_train_task = PythonOperator(
            task_id = 'p_tuning_train',
            python_callable= p_tuning_training_bcp,
            op_kwargs= {"org":org_, "ace": ace_, "team": team_},
            trigger_rule=TriggerRule.ONE_SUCCESS,
            dag = dag)
    
    wait_task = PythonOperator(
            task_id = 'wait_for_job_completion',
            python_callable= wait_for_job_completion,
            op_kwargs= {"org":org_, "team": team_},
            dag = dag)

token_task >> p_tuning_train_task >> wait_task
