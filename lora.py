from ngc_requests import find_file_in_workspace, ngc_job_request, wait_for_job_completion

def lora_training_bcp(ti, ngc_api_key, org, ace, team=None):
      '''Launches a LoRA training job on BCP via NeMo Framework Training container'''
      
      #get workspace ids
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #avoid retraining if job has already ran and we have our LoRA model
      lora_model_exists=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'lora_gpt3_5b.nemo')
      if lora_model_exists:
            return

      #get the base LLM from upstream Airflow tasks
      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('GPT pretraining not implemented. Consider rerunning with a pretrained .nemo checkpoint.')

      #ngc job parameters
      job_name = "airflow_lora_gpt3_5b_train"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1 #single node
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      #note: these settings are currently for GPT 5B BF16 TP2
      job_command = f"python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
            name=lora_gpt3_5b \
            trainer.devices=2 \
            trainer.accelerator=gpu \
            trainer.num_nodes=1 \
            trainer.precision=bf16 \
            trainer.max_epochs=4 \
            trainer.max_steps=100 \
            trainer.log_every_n_steps=10 \
            trainer.val_check_interval=1.0 \
            trainer.gradient_clip_val=1.0 \
            exp_manager.explicit_log_dir=/mount/tuning_workspace/training_info \
            exp_manager.exp_dir=/mount/tuning_workspace/peft_lora \
            exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
            model.tensor_model_parallel_size=2 \
            model.pipeline_model_parallel_size=1 \
            model.global_batch_size=32 \
            model.micro_batch_size=8 \
            model.data.train_ds.num_workers=0 \
            model.data.validation_ds.num_workers=0 \
            model.data.train_ds.concat_sampling_probabilities=[1.0] \
            model.data.validation_ds.names=['squad_val'] \
            model.peft.peft_scheme='lora' \
            model.answer_only_loss=True \
            model.restore_from_path=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
            model.data.train_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_train.jsonl] \
            model.data.validation_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_val.jsonl]\
            +model.data.chat=False"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=300, team=team)

      return job_response


def lora_inference_bcp(ti, ngc_api_key, org, ace, team=None):
      '''Launches a LoRA inference job on BCP via inference/eval scripts in
      NeMo Framework Training container (yes - training container :) )'''
      
      #get workspace id
      gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
      tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

      #avoid rerunning inference if we already have inference results for LoRA
      lora_inference_txt_exists=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'lora_gpt3_5b_inference.txt')
      if lora_inference_txt_exists:
            return

      #get the base LLM from upstream Airflow tasks - LoRA model produced aftern training
      #isn't merged, so we still have to include the base LLM
      pretrain_decision=ti.xcom_pull(task_ids='get_base_model')
      if pretrain_decision=='download_nemo_checkpoint':
            _,_, gpt_base_model_name=ti.xcom_pull(task_ids='download_nemo_checkpoint') #.nemo file
      else:
            raise NotImplementedError('GPT pretraining not implemented. Consider rerunning with a pretrained .nemo checkpoint.')

      #ngc job parameters
      job_name = "airflow_lora_gpt3_5b_inference"
      ace_instance = "dgxa100.80g.2.norm"
      ace_name = ace
      docker_image = f"{org}/nemofw-training:23.07-py3"
      replica_count = 1
      workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                  {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
      
      #note: these settings are currently for GPT 5B BF16 TP2
      job_command = f"python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
            trainer.devices=2 \
            trainer.precision=bf16 \
            model.restore_from_path=/mount/gpt_workspace/gpt_models/{gpt_base_model_name} \
            model.tensor_model_parallel_size=2 \
            model.global_batch_size=16 \
            model.micro_batch_size=4 \
            model.peft.peft_scheme='lora' \
            model.peft.restore_from_path=/mount/tuning_workspace/training_info/checkpoints/lora_gpt3_5b.nemo \
            model.data.test_ds.file_names=[/mount/tuning_workspace/SQuAD/v1.1/squad_test.jsonl] \
            model.data.test_ds.names=['my_test_set'] \
            model.data.test_ds.global_batch_size=4 \
            model.data.test_ds.micro_batch_size=1 \
            model.data.test_ds.tokens_to_generate=30 \
            inference.greedy=True \
            inference.outfile_path=/mount/tuning_workspace/training_info/lora_gpt3_5b_inference.txt"
      
      #send ngc job request
      job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

      #wait for job to complete on BCP before allowing airflow to "finish" task
      final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=30, team=team)

      return job_response