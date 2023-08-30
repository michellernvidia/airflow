from ngc_requests import find_file_in_workspace, ngc_job_request, wait_for_job_completion

    
def p_tuning_training_bcp(ti, ngc_api_key, org, ace, team=None):
      '''Launches a p-tuning training job on BCP via NeMo Framework Training container'''

      #get workspace id
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #avoid retraining if job has already ran and we have our p-tuned model
      p_tuned_model_exists=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'p_tuned_gpt3_5b.nemo')
      if p_tuned_model_exists:
            return

      #get the base LLM from upstream Airflow tasks
      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('GPT pretraining not implemented. Consider rerunning with a pretrained .nemo checkpoint.')

      #ngc job parameters
      job_name = "airflow_p_tuning_gpt3_5b_train"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # Set-up for GPT 5B TP 2 BF16
      job_command = f"python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
            prompt_learning=gpt3/squad \
            stages=[prompt_learning] \
            cluster_type=bcp \
            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
            data_dir=/mount/tuning_workspace \
            base_results_dir=/mount/tuning_workspace/p_tuning_results \
            prompt_learning.run.model_train_name=gpt3_5b \
            prompt_learning.trainer.devices=2 \
            prompt_learning.model.language_model_path=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
            prompt_learning.model.nemo_path=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_gpt3_5b.nemo \
            prompt_learning.model.tensor_model_parallel_size=2 \
            >> /results/prompt_learning_gpt3_log.txt 2>&1"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=300, team=team)

      return job_response

    
def p_tuning_inference_bcp(ti, ngc_api_key, org, ace, team=None):
      '''Launches a p-tuning inference job on BCP via NeMo Framework Training container'''
      
      #get workspace ids
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #check if we already have p-tuning inference results in our workspace
      ptuning_inference_results_exist=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'p_tuned_gpt3_5b_inference.txt')
      if ptuning_inference_results_exist:
            return

      #get the base LLM from upstream Airflow tasks
      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('GPT pretraining not implemented. Consider rerunning with a pretrained .nemo checkpoint.')

      #ngc job parameters
      job_name = "airflow_p_tuning_gpt3_5b_inference"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # Set-up for GPT 5B TP 2 BF16 with SQuAD dataset
      job_command = f"cd ../; python3 opt/NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py \
                        --config-path=/opt/NeMo/examples/nlp/language_modeling/conf/ \
                        --config-name=megatron_gpt_prompt_learning_inference.yaml \
                        gpt_model_file=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
                        virtual_prompt_model_file=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_gpt3_5b.nemo \
                        data_paths=['/mount/tuning_workspace/SQuAD/v1.1/squad_test.jsonl'] \
                        pred_file_path=/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_gpt3_5b_inference.txt \
                        trainer.devices=2 \
                        tensor_model_parallel_size=2 \
                        pipeline_model_parallel_size=1"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=300, team=team)

      return job_response