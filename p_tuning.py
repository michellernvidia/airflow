import time
from ngc_requests import *

    
def p_tuning_training_bcp(ti, ngc_api_key, org, ace, team=None):
      return

      #get workspace id
      # _, workspace_id = ti.xcom_pull(task_ids='download_nemo_checkpoint')
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('Need to retrieve name of checkpoint from pretraining GPT step.') #TO DO

      #ngc job parameters
      job_name = "p_tuning_train_gpt5b_airflow"
      ace_instance = "dgxa100.80g.4.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # To Do: Currentnly configured for GPT 5B - fix!! make general
      job_command = f"python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
                            prompt_learning=gpt3/squad \
                            stages=[prompt_learning] \
                            cluster_type=bcp \
                            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
                            data_dir=/mount/tuning_workspace \
                            base_results_dir=/mount/tuning_workspace/p_tuning_results \
                            prompt_learning.run.model_train_name=gpt3_5b \
                            prompt_learning.trainer.devices=4 \
                            prompt_learning.model.language_model_path=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
                            prompt_learning.model.nemo_path=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_5b.nemo \
                            prompt_learning.model.tensor_model_parallel_size=2 \
                            >> /results/prompt_learning_gpt3_log.txt 2>&1"
                        #     prompt_learning.model.virtual_prompt_style='p-tuning' \

      
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

    
def p_tuning_inference_bcp(ti, ngc_api_key, org, ace, team=None):
      
      #get workspace id
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('Need to retrieve name of checkpoint from pretraining GPT step.') #TO DO

      #ngc job parameters
      job_name = "p_tuning_inference_gpt5b_airflow"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.05-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # To Do: Currentnly configured for GPT 5B - fix!! make general
      #have to edit virtual_prompt_model_file name
      job_command = f"cd ../; python3 opt/NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py \
                        --config-path=/opt/NeMo/examples/nlp/language_modeling/conf/ \
                        --config-name=megatron_gpt_prompt_learning_inference.yaml \
                        gpt_model_file=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
                        virtual_prompt_model_file=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_5b.nemo \
                        data_paths=['/mount/tuning_workspace/SQuAD/v1.1/squad_test.jsonl'] \
                        pred_file_path=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_5b_inference_results.txt \
                        trainer.devices=2 \
                        tensor_model_parallel_size=2 \
                        pipeline_model_parallel_size=1"
      
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