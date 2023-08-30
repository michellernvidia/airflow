from ngc_requests import find_file_in_workspace, ngc_job_request, wait_for_job_completion

def squad_metric_eval(ti, ngc_api_key, org, ace,tuning_method, team=None):
    '''Launches a job on BCP using the NeMo Framework training container to 
    quantify model performance by calculating F1 scores and exact_match metrics on the SQuAD results.'''

    #get workspace id
    tuning_workspace_id = ti.xcom_pull(task_ids='create_tuning_workspace')

    #ngc job parameters
    job_name = f"airflow_{tuning_method}_gpt3_5b_squad_metric_eval"
    ace_instance = "dgxa100.80g.1.norm"
    ace_name = ace
    docker_image = f"{org}/nemofw-training:23.07-py3"
    replica_count = 1
    workspaces=[{"id":tuning_workspace_id, "mount": "/mount/tuning_workspace"}]

    #paths to inference results for each model
    if tuning_method == 'sft':
        inference_preds = '/mount/tuning_workspace/sft_launcher_results/gpt3_5b_sft/squad/results/sft_gpt3_5b_inference.jsonl'
        split_string= 'Assistant:'
        answer_field= 'output'
    elif tuning_method == 'lora':
        inference_preds = '/mount/tuning_workspace/SQuAD/v1.1/squad_test_ground_truth.jsonl'
        split_string= 'Assistant:'
        answer_field= 'output'
    elif tuning_method=='p_tuning':
        inference_preds = '/mount/tuning_workspace/p_tuning_results/gpt3_5b/prompt_learning_squad/results/p_tuned_gpt3_5b_inference.txt'
        split_string= 'Answer:'
        answer_field= 'answer'

    #command to merge LoRA adapter weights with base LLM on BCP (DGX Cloud)
    job_command=f"python3 /opt/NeMo/scripts/metric_calculation/squad_metric_calc.py \
                --ground-truth /mount/tuning_workspace/SQuAD/v1.1/squad_test_ground_truth.jsonl \
                --preds {inference_preds} \
                --split-string {split_string} \
                --answer-field {answer_field}"
    
    job_response = ngc_job_request(ti, ngc_api_key, org, job_name, ace_instance, ace_name, docker_image, \
                                     replica_count, workspaces, job_command, team=team)

    #wait for job to complete on BCP before allowing airflow to "finish" task
    final_job_status = wait_for_job_completion(ti, ngc_api_key, org, job_response, wait_time=30, team=team)
    
    return job_response