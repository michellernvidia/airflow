import time
from ngc_requests import *

    
def p_tuning_training_bcp(ti, ngc_api_key, org, ace, team=None):

      #TODO: NEED TO MODIFY NGC_JOB_REQUESTS TO ACCEPT MOUNTING MORE THAN 1 WORKSPACE
      
      #get workspace id
      # _, workspace_id = ti.xcom_pull(task_ids='download_nemo_checkpoint')
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      
      #ngc job parameters
      job_name = "p_tuning_train_gpt5b_airflow"
      ace_instance = "dgxa100.80g.4.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      job_command = "python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
                            prompt_learning=gpt3/squad \
                            stages=[prompt_learning] \
                            cluster_type=bcp \
                            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
                            data_dir=/mount/tuning_workspace \
                            base_results_dir=/mount/tuning_workspace/results \
                            prompt_learning.run.model_train_name=gpt3_5b \
                            prompt_learning.trainer.devices=4 \
                            prompt_learning.model.language_model_path=/mount/gpt_workspace/gpt_models/nemo_gpt5B_bf16_tp2.nemo \
                            prompt_learning.model.tensor_model_parallel_size=2 \
                            >> /results/prompt_learning_gpt3_log.txt 2>&1"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti,
                                                 ngc_api_key,
                                                 org, 
                                                 job_response, 
                                                 wait_time=300, 
                                                 team=team)

      return job_response