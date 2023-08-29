import time
from ngc_requests import *

'''Python file containing functions relevant to the path where we 
load in a nemo checkpoint and validate its successful download'''

def download_nemo_checkpoint(ti, ngc_api_key, org, ace, nemo_ckpt_file, team=None):

      #get workspace id
      workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')

      # check if .nemo file already exists in ngc wksp to avoid redownloading
      checkpoint_exists=find_file_in_workspace(ngc_api_key, org, workspace_id, nemo_ckpt_file)
      if checkpoint_exists==True:
            return None, workspace_id, nemo_ckpt_file
      
      #ngc job parameters
      job_name = "airflow_download_gpt_nemo_ckpt"
      ace_instance = "dgxa100.80g.1.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{'id': workspace_id, 'mount': "/mount/gpt_workspace"}]

      job_command = f"cd ../; cd /mount/gpt_workspace/; mkdir gpt_models; cd gpt_models;\
                    wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/{nemo_ckpt_file}"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)
      
      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti,
                                                 ngc_api_key, 
                                                 org, 
                                                 job_response, 
                                                 wait_time=15, 
                                                 team=team)

      return job_response, workspace_id, nemo_ckpt_file
 