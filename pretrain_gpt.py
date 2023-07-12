'''Python file containing functions relevant to training a gpt model from scratch and then
performing p-tuning on it'''

import os, json, base64, requests, time
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable

from ngc_requests import create_workspace, ngc_job_request, ngc_job_status


# NEEDS TO BE RUN + TESTED ON AIRFLOW
def download_pile_dataset(ti, org, ace, team=None):
     return
     
    #  #create workspace to download the pile dataset into
    #  pile_data_workspace = create_workspace()
    #  workspace_id = pile_data_workspace['id']

    #  #job parameters
    #  job_name = "download_the_pile_dataset"
    #  ace_instance = "dgxa100.80g.8.norm"
    #  ace_name = ace
    #  docker_image = f"{org}/nemofw-training:23.04.1-py3" 
    #  replica_count = 4
    #  workspace_mount_path = "/mount_workspace"
    #  job_command = "\
    #                 set -x && \
    #                 mkdir -p /mount_workspace/data/bpe && \
    #                 wget https://huggingface.co/gpt2/resolve/main/vocab.json -O /mount_workspace/data/bpe/vocab.json && \
    #                 wget https://huggingface.co/gpt2/resolve/main/merges.txt -O /mount_workspace/data/bpe/merges.txt && \
    #                 python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
    #                 cluster_type=bcp \
    #                 stages=[data_preparation] \
    #                 launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
    #                 data_dir=/mount_workspace/data \
    #                 base_results_dir=/mount_workspace/results \
    #                 data_preparation.run.node_array_size=\${NGC_ARRAY_SIZE} \
    #                 data_preparation.the_pile_url=https://the-eye.eu/public/AI/pile/train/ \
    #                 data_preparation.file_numbers='0-29' \
    #                 data_preparation.rm_downloaded=True \
    #                 data_preparation.rm_extracted=True \
    #                 data_preparation.tokenizer_type=GPT2BPETokenizer"
     
    #  total_runtime = "10h"
    #  array_type = "PYTORCH"
    #  multinode=True

    #  job_response = ngc_job_request(ti, org, job_name, ace_instance, ace_name, docker_image, replica_count, \
    #                 workspace_mount_path, workspace_id, job_command, team, multinode, array_type, total_runtime)

    #  return job_response

# NEEDS TO BE RUN + TESTED ON AIRFLOW
def train_gpt_model(ti, org, ace, team=None):
     return