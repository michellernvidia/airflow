'''File with branching functions for NeMo Airflow DAG'''

def get_base_model(ti, pretrain_decision):
    if pretrain_decision == "False":
        return 'download_nemo_checkpoint'
    else:
        return 'download_pile_dataset'
    
def choose_tuning_method(ti, method):
    if method == 'p_tuning':
        return 'p_tuning_train'
    elif method == 'lora':
        return 'LoRA_train'
    elif method == 'sft':
        return 'SFT_train'
    
def choose_inference_method(ti, interactive, method):
    if method == 'lora':
        if interactive:
            return 'Merge_Adapter_Weights'
        else:
            return 'LoRA_inference'
    elif method == 'sft':
        if interactive:
            return ''
        else:
            return 'SFT_inference'
    elif method == 'p_tuning':
        if interactive:
            return ''
        else:
            return 'p_tuning_inference'
