'''Python file containing functions relevant to training a gpt model from scratch and then
performing p-tuning on it'''

import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable

from ngc_requests import create_workspace, ngc_job_request, wait_for_job_completion
    

def download_missing_files():
    return

# NEEDS TO BE RUN + TESTED ON AIRFLOW
def download_pile_dataset(ti, ngc_api_key, org, ace, team=None):
     
     #create workspace to download the pile dataset into
    #  workspace_name = 'the_pile_dataset_airflow'
    #  pile_data_workspace = create_workspace(ti, ngc_api_key, org, ace, workspace_name)
    #  workspace_id = pile_data_workspace['workspace']['id']

     workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')

     #job parameters
     job_name = "download_the_pile_dataset_airflow"
     ace_instance = "dgxa100.80g.8.norm"
     ace_name = ace
     docker_image = f"{org}/nemofw-training:23.07-py3" 
     replica_count = 4 #number of nodes
     total_runtime = "10h"
     array_type = "PYTORCH"
     multinode=True
     workspaces=[{'id': workspace_id, 'mount': '/mount_workspace'}]
     job_command = "\
                    set -x && \
                    mkdir -p /mount_workspace/data/bpe && \
                    wget https://huggingface.co/gpt2/resolve/main/vocab.json -O /mount_workspace/data/bpe/vocab.json && \
                    wget https://huggingface.co/gpt2/resolve/main/merges.txt -O /mount_workspace/data/bpe/merges.txt && \
                    python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
                    cluster_type=bcp \
                    stages=[data_preparation] \
                    launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
                    data_dir=/mount_workspace/data \
                    base_results_dir=/mount_workspace/results \
                    data_preparation.run.node_array_size=${NGC_ARRAY_SIZE} \
                    data_preparation.the_pile_url=https://the-eye.eu/public/AI/pile/train/ \
                    data_preparation.file_numbers='0-29' \
                    data_preparation.rm_downloaded=True \
                    data_preparation.rm_extracted=True \
                    data_preparation.tokenizer_type=GPT2BPETokenizer"
     
     job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
                    workspaces, job_command, team, multinode, array_type, total_runtime)
     
     #get a job status update from ngc every hour
     final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=3600, team=team)

     return job_response, workspace_id

# NEEDS TO BE RUN + TESTED ON AIRFLOW
def train_gpt_model(ti, ngc_api_key, org, ace, team=None):
     
     #get the pile dataset workspace info
    #  _, workspace_id = ti.xcom_pull(task_ids='download_pile_dataset')
     workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')

     job_name="airflow-gpt3-training-5b-bf16"
     ace_instance="dgxa100.80g.8.norm"
     ace_name=ace
     docker_image=f"{org}/nemofw-training:23.07-py3"
     replica_count=4 #number of nodes to run job on
     multinode=True
     array_type="PYTORCH"
     total_runtime="5D"
     workspaces=[{'id': workspace_id, 'mount': '/mount_workspace'}]
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
            training.trainer.num_nodes=${NGC_ARRAY_SIZE} \
            training.model.tokenizer.vocab_file=/mount_workspace/data/bpe/vocab.json \
            training.model.tokenizer.merge_file=/mount_workspace/data/bpe/merges.txt \
            > >(tee -a /results/train_log.log) \
            2> >(tee -a /results/train_stderr.log >&2) && \
            rsync -P -rvh /mount_workspace/results /results"
     
     job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
                    workspaces, job_command, team, multinode, array_type, total_runtime)
     #get a job status update from ngc twice a day
     #NOTE: we should probably make this an external dag because of the risk of leaving the dag hanging
     #for too long while we wait for the last status update
     interval = 60 * 60 * 12 
     final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=interval, team=team)

     return job_response