from ngc_requests import find_file_in_workspace, ngc_job_request, wait_for_job_completion

def sft_training_bcp(ti, ngc_api_key, org, ace, team=None):

      #get workspace id
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #avoid retraining if job has already ran and we have our sft model
      sft_model_exists=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'megatron_gpt3_squad.nemo')
      if sft_model_exists:
            return
      
      #get the base LLM from upstream Airflow tasks
      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('GPT pretraining not implemented. Consider rerunning with a pretrained .nemo checkpoint.')

      #ngc job parameters
      job_name = "airflow_sft_gpt3_5b_train"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # Configured for GPT3 5B BF16 TP2
      job_command = f"python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
            fine_tuning=gpt3/squad \
            stages=[fine_tuning] \
            cluster_type=bcp \
            launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts \
            data_dir=/mount/tuning_workspace/SQuAD/v1.1 \
            base_results_dir=/mount/tuning_workspace/sft_launcher_results \
            fine_tuning.trainer.devices=2 \
            fine_tuning.run.model_train_name=gpt3_5b_sft \
            fine_tuning.model.restore_from_path=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
            fine_tuning.model.tensor_model_parallel_size=2 \
            fine_tuning.model.pipeline_model_parallel_size=1 \
            fine_tuning.model.global_batch_size=32 \
            fine_tuning.model.micro_batch_size=4 \
            fine_tuning.model.data.train_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_train.jsonl] \
            fine_tuning.model.data.train_ds.concat_sampling_probabilities=[1.0] \
            fine_tuning.model.data.validation_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_val.jsonl] \
            fine_tuning.model.data.validation_ds.names=[squad_validation_data] \
            fine_tuning.model.data.validation_ds.metric.name=loss \
            fine_tuning.model.data.test_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_test.jsonl]"

      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=300, team=team)

      return job_response


def sft_inference_bcp(ti, ngc_api_key, org, ace, team=None):

      #get workspace id
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #check if we already have SFT inference results in our workspace
      sft_inference_results_exist=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'sft_gpt3_5b_inference.jsonl')
      if sft_inference_results_exist:
            return

      #ngc job parameters
      job_name = "airflow_sft_gpt3_5b_inference"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      # GPT 5B BF16 TP2
      job_command=f"python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
            model.restore_from_path=/mount/tuning_workspace/sft_launcher_results/gpt3_5b_sft/squad/results/checkpoints/megatron_gpt3_squad.nemo \
            model.peft.restore_from_path=null \
            trainer.devices=2 \
            model.data.test_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_test.jsonl] \
            model.data.test_ds.names=['squad_test'] \
            model.data.test_ds.global_batch_size=32 \
            model.data.test_ds.micro_batch_size=4 \
            model.data.test_ds.tokens_to_generate=30 \
            inference.greedy=True \
            inference.outfile_path=/mount/tuning_workspace/sft_launcher_results/gpt3_5b_sft/squad/results/sft_gpt3_5b_inference.jsonl"

      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=300, team=team)

      return job_response