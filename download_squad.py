import time
from ngc_requests import *


def download(ti, ngc_api_key, org, ace, team, workspace_id):
    
    #ngc job parameters
    job_name = "download_squad_airflow"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.05-py3"
    replica_count = 1
    workspaces=[{'id': workspace_id, 'mount': '/mount/tuning_workspace'}]
    job_command ="python3 -c \"from nemo_launcher.utils.data_utils.prepare_squad import prepare_squad_for_fine_tuning; \
                prepare_squad_for_fine_tuning( '/mount/tuning_workspace/SQuAD')\""
    
    #send ngc job request
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                    replica_count, workspaces, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti,
                                                ngc_api_key,
                                                org, 
                                                job_response, 
                                                wait_time=60, 
                                                team=team)

    return job_response

def preprocess(ti, ngc_api_key, org, ace, team, workspace_id, tuning_method):

    #ngc job parameters
    job_name = "preprocess_squad_airflow"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.05-py3"
    replica_count = 1
    workspaces=[{'id': workspace_id, 'mount': '/mount/tuning_workspace'}]
    
    
    if tuning_method.lower() in ['sft', 'lora']:
        job_command = "wget --directory-prefix=/mount/tuning_workspace https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py; \
                python3 /mount/tuning_workspace/prompt_learning_squad_preprocessing.py --sft-format --data-dir /mount/tuning_workspace/SQuAD/v1.1"
    else: #format for p-tuning
        job_command = "wget --directory-prefix=/mount/tuning_workspace https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py; \
                python3 /mount/tuning_workspace/prompt_learning_squad_preprocessing.py --data-dir /mount/tuning_workspace/SQuAD/v1.1"
    
    #send ngc job request
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                    replica_count, workspaces, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti,
                                                ngc_api_key,
                                                org, 
                                                job_response, 
                                                wait_time=60, 
                                                team=team)

    return job_response

def get_squad_dataset(ti, ngc_api_key, org, ace, team, tuning_method):

    #get NGC workspace id where we plan to download squad into
    workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

    #check if squad files exist in workspace
    squad_files=['squad_train.jsonl', 'squad_val.jsonl', 'squad_test.jsonl', 'squad_test_ground_truth.jsonl']
    num_existing_squad_files=0
    for file in squad_files:
        file_exists=find_file_in_workspace(ngc_api_key, org, workspace_id, file)
        num_existing_squad_files+=int(file_exists)
    
    #all 4 files already exist
    if num_existing_squad_files==4:
        return
    
    #dataset does not exist - download and preprocess squad
    download_job_response = download(ti, ngc_api_key, org, ace, team, workspace_id)
    preprocess_job_response = preprocess(ti, ngc_api_key, org, ace, team, workspace_id, tuning_method)
    return