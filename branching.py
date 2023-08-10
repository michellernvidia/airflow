

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