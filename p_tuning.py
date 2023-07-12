import time
from ngc_requests import *


def p_tuning(ti, pretrain_decision):
    if pretrain_decision == "False":
        return 'download_nemo_checkpoint'
    else:
        return 'download_pile_dataset'
    
def p_tuning_training_bcp(ti, org, ace, team=None):
      
      #get workspace id
      _, workspace_id = ti.xcom_pull(task_ids='download_nemo_checkpoint')
      
      #ngc job parameters
      job_name = "p_tuning_train_gpt5b_airflow"
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

      #wait for job to complete on BCP before allowing airflow to "finish" task
      job_id = job_response['job']['id']
      job_status = ngc_job_status(ti, org, job_id)
      while job_status != 'FINISHED_SUCCESS' and job_status != 'FAILED':
            time.sleep(300) #increase wait time to 5 mins
            job_status = ngc_job_status(ti, org, job_id)
            print(job_status)

      return job_response