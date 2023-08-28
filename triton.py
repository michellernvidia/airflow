import time
from ngc_requests import *


def merge_lora_weights(ti, ngc_api_key, org, ace, team=None):
    #get workspace id
    gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
    tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

    #avoid merging if file already exists
    merged_model_exists=find_file_in_workspace(ngc_api_key, org, tuning_workspace_id, 'lora_gpt_5B_merged.nemo')
    if merged_model_exists:
        return
    
    #ngc job parameters
    job_name = f"airflow_lora_merge_weights"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.07-py3"
    replica_count = 1
    workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]

    #currently set up to use GPT 5B - will need to generalize
    job_command="python3 /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py \
                trainer.devices=2 \
                trainer.precision=bf16 \
                tensor_model_parallel_size=2 \
                pipeline_model_parallel_size=1 \
                gpt_model_file=/mount/gpt_workspace/gpt_models/nemo_gpt5B_bf16_tp2.nemo \
                lora_model_path=/mount/tuning_workspace/training_info/checkpoints/lora_gpt_airflow_tuning.nemo \
                merged_model_path=/mount/tuning_workspace/training_info/checkpoints/lora_gpt_5B_merged.nemo"
    
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=30, team=team)
    
    return job_response


def create_triton_model_repository(ti, ngc_api_key, org, ace, team=None, method=None):
    '''Converts .nemo file into faster transformer + creates the model repository necessary to 
    serve the model through Triton inference server'''
    
    #get workspace id
    gpt_workspace_id = ti.xcom_pull(task_ids='create_gpt_workspace')
    tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')
    
    #ngc job parameters
    job_name = f"airflow_triton_inference_{method}"
    ace_instance = "dgxa100.80g.2.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.07-py3"
    replica_count = 1
    workspaces=[{"id":gpt_workspace_id, "mount": "/mount/gpt_workspace"}, 
                {"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]
    

    if method=="p_tuning":
        nemo_file_path="nemo_gpt5B_bf16_tp2.nemo"
    elif method == "lora":
        nemo_file_path="/mount/tuning_workspace/training_info/checkpoints/lora_gpt_5B_merged.nemo"
        model_train_name="lora_gpt5B"
    elif method == "sft":
        nemo_file_path="megatron_gpt3_squad.nemo"

    job_command = f"\
                    bash -c 'export PYTHONPATH=/opt/FasterTransformer:\${PYTHONPATH} && \
                    cd /opt && \
                    python3 /opt/FasterTransformer/examples/pytorch/gpt/utils/nemo_ckpt_convert.py \
                        --in-file {nemo_file_path} \
                        --infer-gpu-num 2 \
                        --saved-dir /mount/tuning_workspace/model_repository/{model_train_name} \
                        --weight-data-type fp16 \
                        --load-checkpoints-to-cpu 0 && \
                    python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/nemo_launcher/collections/export_scripts/prepare_triton_model_config.py \
                        --model-train-name {model_train_name} \
                        --template-path /opt/fastertransformer_backend/all_models/gpt/fastertransformer/config.pbtxt \
                        --ft-checkpoint /mount/tuning_workspace/model_repository/{model_train_name}/2-gpu \
                        --config-path /mount/tuning_workspace/model_repository/{model_train_name}/config.pbtxt \
                        --max-batch-size 256 \
                        --tensor-model-parallel-size 2 \
                        --pipeline-model-parallel-size 1 \
                        --data-type fp16' "
    
    #send ngc job request
    job_response = ngc_job_request(
        ti,
        ngc_api_key, 
        org,
        job_name,
        ace_instance,
        ace_name,
        docker_image,
        replica_count,
        workspaces,
        job_command,
        team=team)
    
    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(
        ti, 
        ngc_api_key, 
        org, 
        job_response, 
        wait_time=15, 
        team=team)

    return job_response

def launch_triton_server():
    return
