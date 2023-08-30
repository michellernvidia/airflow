# File with branching functions for NeMo Airflow DAG

def get_base_model(ti, pretrain_decision):
    '''Branching decision for base LLM pretraining task in Airflow'''

    if pretrain_decision == "False":
        return 'download_nemo_checkpoint'
    else:
        return 'download_pile_dataset'


def choose_tuning_method(ti, method):
    '''Branching decision for selecting tuning method in Airflow'''

    if method == 'p_tuning':
        return 'p_tuning_train'
    elif method == 'lora':
        return 'LoRA_train'
    elif method == 'sft':
        return 'SFT_train'


def choose_inference(ti, interactive, method):
    '''Branching decision for selecting inference strategy to run on tuned model in Airflow'''
    
    if method == 'lora':
        if interactive:
            return 'triton_inference.merge_lora_adapter_weights'
        else:
            return 'nemo_script_inference.LoRA_inference_script'
    elif method == 'p_tuning':
        if interactive:
            return 'triton_inference.create_triton_model_repository'
        else:
            return 'nemo_script_inference.p_tuning_inference_script'
    elif method == 'sft':
        if interactive:
            return 'triton_inference.create_triton_model_repository'
        else:
            return 'nemo_script_inference.SFT_inference_script'
