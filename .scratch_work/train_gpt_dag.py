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


def train_gpt_model(ti, ngc_api_key, org, ace, team=None):
     
     #get the pile dataset workspace info
     workspace_id = "dL_-JJeZQ9KbnWB6u0QFhw"

     job_name="airflow-gpt3-training-5b-bf16-standalone-task"
     ace_instance="dgxa100.80g.8.norm"
     ace_name=ace
     docker_image=f"{org}/nemofw-training:23.05-py3"
     replica_count=4 #number of nodes to run job on
     workspace_mount_path="/mount_workspace"
     multinode=True
     array_type="PYTORCH"
     total_runtime="5D"
     job_command="\
            set -x && \
            python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
            cluster_type=bcp \
            stages=[training] \
            training=gpt3/5b \
            training_config=gpt3/5b \
            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
            data_dir=/mount_workspace/data \
            base_results_dir=/mount_workspace/results \
            training.run.time_limit=\"5-00:00:00\" \
            training.trainer.max_time=\"4:23:30:00\" \
            training.trainer.num_nodes=\${NGC_ARRAY_SIZE} \
            training.model.tokenizer.vocab_file=/mount_workspace/data/bpe/vocab.json \
            training.model.tokenizer.merge_file=/mount_workspace/data/bpe/merges.txt \
            > >(tee -a /results/train_log.log) \
            2> >(tee -a /results/train_stderr.log >&2) && \
            rsync -P -rvh /mount_workspace/results /results"
     
     job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
                    workspace_mount_path, workspace_id, job_command, team, multinode, array_type, total_runtime)
     #get a job status update from ngc twice a day
     #NOTE: we should probably make this an external dag because of the risk of leaving the dag hanging
     #for too long while we wait for the last status update
     interval = 60 * 60 * 12 
     final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=interval, team=team)

     return job_response, final_job_status


#this is really just to have an upstream task
def confirm_job_finished(ti):
     _, final_job_status = ti.xcom_pull(task_ids='train_gpt_model')
     print(final_job_status)
     return

## Define DAG + Tasks
with DAG(
         "PRETRAINING_GPT_ONLY", 
         schedule_interval='@once',
         start_date=datetime(2022, 1, 1),
         catchup=False
    ) as dag: 

    train_gpt_task = PythonOperator(
            task_id = 'train_gpt_model',
            python_callable= train_gpt_model,
            op_kwargs= {"ngc_api_key": key_, "org":org_, "ace": ace_, "team": team_},
            dag = dag)
    
    confirm_end_task = PythonOperator(
            task_id = 'confirm_job_finished',
            python_callable= confirm_job_finished,
            dag = dag)

train_gpt_task >> confirm_end_task


