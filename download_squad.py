import time
from ngc_requests import *


def download(ti, ngc_api_key, org, ace, team):

    #get workspace id
    _, workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')
    
    #ngc job parameters
    job_name = "download_squad_dataset"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.05-py3"
    replica_count = 1
    workspace_mount_path = "/mount/tuning_workspace"
    job_command ="python3 -c \"from nemo_launcher.utils.data_utils.prepare_squad import prepare_squad_for_fine_tuning; \
                prepare_squad_for_fine_tuning( '/mount/tuning_workspace/SQuAD')\""
    
    #send ngc job request
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                    replica_count, workspace_mount_path, workspace_id, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti,
                                                ngc_api_key,
                                                org, 
                                                job_response, 
                                                wait_time=60, 
                                                team=team)

    return job_response

def preprocess(ti, ngc_api_key, org, ace, team, tuning_method):

    #get workspace id
    _, workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

    #ngc job parameters
    job_name = "squad_data_preprocessing"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.05-py3"
    replica_count = 1
    workspace_mount_path = "/mount/tuning_workspace"
    
    
    if tuning_method.lower() in ['sft', 'lora']:
        job_command = "wget --directory-prefix=/mount/tuning_workspace https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py; \
                python3 /mount/tuning_workspace/prompt_learning_squad_preprocessing.py --sft-format --data-dir /mount/tuning_workspace/SQuAD/v1.1"
    else:
        job_command = "wget --directory-prefix=/mount/tuning_workspace https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py; \
                python3 /mount/tuning_workspace/prompt_learning_squad_preprocessing.py --data-dir /mount/tuning_workspace/SQuAD/v1.1"
    
    #send ngc job request
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                    replica_count, workspace_mount_path, workspace_id, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti,
                                                ngc_api_key,
                                                org, 
                                                job_response, 
                                                wait_time=60, 
                                                team=team)

    return job_response

def get_squad_dataset(ti, ngc_api_key, org, ace, team, tuning_method):
    download_job_response = download(ti, ngc_api_key, org, ace, team)
    preprocess_job_response = preprocess(ti, ngc_api_key, org, ace, team, tuning_method)
    return
                      