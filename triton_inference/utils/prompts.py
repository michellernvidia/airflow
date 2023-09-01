import numpy as np
import torch
from utils import PROMPT_TYPE
from typing import Any, List, Tuple


def _convert_to_fp16(prompt_table: Any, taskname: str) -> Any:
    """
    Convert the prompt embeddings to FP16.

    The weights of the prompt embeddings need to be converted from BF16 to
    FP16 if not already done as Triton currently only supports FP16 and FP32.
    Most p-tuned models will use BF16 by default and will not work without
    converting the data type.

    Args:
        prompt_table: The extracted prompt table from the model checkpoint.

    Returns
        Returns the converted prompt table in FP16 format.
    """
    fp16_dtype = prompt_table[f'prompt_table.{taskname}.prompt_embeddings.weight'].to(torch.float16)
    prompt_table[f'prompt_table.{taskname}.prompt_embeddings.weight'] = fp16_dtype
    return prompt_table


def build_request_prompts(
        model_weights: str,
        taskname: str
    ) -> Tuple[np.array, Any, np.array]:
    """
    Create inputs to send to the inference server which include the prompt
    embeddings from the tuned model.

    Args:
        model_weights: The path to the model weights file from the p-tuned model.
        taskname: The name of the task for the specified prompt.

    Returns:
        A tuple of the prompt lengths, prompt embedding, and prompt type for
        the request.
    """
    prompt_table = torch.load(model_weights, map_location=torch.device('cpu'))['prompt_table']
    prompt_table = _convert_to_fp16(prompt_table, taskname)
    prompt_embedding = prompt_table[f'prompt_table.{taskname}.prompt_embeddings.weight']
    prompt_length = prompt_embedding.shape[0]

    request_prompt_lengths = prompt_length * np.ones([1, 1]).astype(np.uint32)
    request_prompt_embedding = np.expand_dims(prompt_embedding, axis=0)
    request_prompt_type = PROMPT_TYPE * np.ones([1, 1]).astype(np.uint32)
    return request_prompt_lengths, request_prompt_embedding, request_prompt_type


def convert_hf_prompt(
        ptuning_mode: bool,
        prompt: str,
        template: dict,
        tokenizer: Any,
        pseudo_tokens: List,
        taskname: str
    ) -> List:
    """
    Convert the input prompt to a list of tokenized IDs including pseudo tokens
    which replace the virtual prompt placeholders in the prompt.

    Args:
        prompt: A string of the requested prompt.
        template: A dictionary of the converted prompt template from the model config.
        tokenizer: The requested tokenizer.
        pseudo_tokens: A list of pseudo tokens to replace the virtual prompt placeholder with.
        taskname: The taskname to generate the prompt for.

    Returns:
        Returns a list of IDs from the tokenized prompt.
    """

    if ptuning_mode:
        #fill in our prompt_template with the actual context, question, and answer text from our PROMPT input
        #SQuAD template for ptuning: '<|VIRTUAL_PROMPT_0|>Context: {context} Question: {question} Answer: {answer}'
        context=prompt.split('\n\n')[0].replace('Context: ', '')
        question=prompt.split('\n\n')[1].replace('Question: ', '')
        answer=prompt.split('\n\n')[2].replace('Answer:', '')
        
        prompt_template=template[taskname]['prompt_template']
        prompt=prompt_template.replace('context', context).replace('question', question).replace('answer', answer)
        prompt = prompt.replace('<|VIRTUAL_PROMPT_0|>', ''.join(pseudo_tokens))
    
    input_tokens = tokenizer.tokenize(prompt)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    return input_ids


def _prompt_source(config: dict) -> str:
    """
    Determine if the model uses a prompt table or prompt encoder.

    Args:
        config: The config file included with the model.

    Returns:
        Returns the virtual prompt style for the specific model.
    """
    # Import locally to avoid the time penalty importing NeMo modules if the
    # NeMo modules aren't used in other paths.
    from nemo.collections.nlp.modules.common import (
        VirtualPromptSource,
        VirtualPromptStyle
    )
    virtual_prompt_style = VirtualPromptStyle(config.virtual_prompt_style)
    
    if virtual_prompt_style == VirtualPromptStyle.P_TUNING:
        virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
    return virtual_prompt_source


def convert_nemo_prompt(
        ptuning_mode: bool,
        prompt: str,
        template: dict,
        tokenizer: Any,
        pseudo_tokens: List,
        taskname: str,
        config: dict = None
    ) -> List:
    """
    Convert the input prompt to a list of tokenized IDs including pseudo tokens
    which replace the virtual prompt placeholders in the prompt. Use the NeMo
    modules to tokenize the prompt and generate a dataset.

    Args:
        prompt: A string of the requested prompt.
        template: A dictionary of the converted prompt template from the model config.
        tokenizer: The requested tokenizer.
        pseudo_tokens: A list of pseudo tokens to replace the virtual prompt placeholder with.
        taskname: The taskname to generate the prompt for.
        config: The config file included with the model.

    Returns:
        Returns a list of IDs from the tokenized prompt.
    """

    if ptuning_mode:
        # Import locally to avoid the time penalty importing NeMo modules if the
        # NeMo modules aren't used in other paths.
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
        
        context=prompt.split('\n\n')[0].replace('Context: ', '')
        question=prompt.split('\n\n')[1].replace('Question: ', '')
        answer=prompt.split('\n\n')[2].replace('Answer:', '')

        data = {'taskname': taskname, 'context': context, 'question': question, 'answer': answer}
        virtual_prompt_source = _prompt_source(config)
        pad_token_id = tokenizer.pad_id if tokenizer.pad_id is not None else tokenizer.unk_id

        dataset = GPTPromptLearningDataset(
            data=[data],
            tokenizer=tokenizer,
            virtual_prompt_source=virtual_prompt_source,
            task_templates=template,
            pseudo_tokens=pseudo_tokens,
            pad_token_id=pad_token_id,
            max_seq_length=2048,
            min_seq_length=1,
            add_bos=False,
            add_eos=False,
            for_train=False
        )

    else:

        raise NotImplementedError()
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
        
        #format prompt into {"input": ...,"output":....}
        input=prompt.split('Answer')[0]
        output=""

        data={"input": f"User:{input}Assistant:", "output": output}

        #got values from fine_tuning gpt3/squad config in nemo-launcher
        dataset=GPTSFTDataset(
            file_path=[data],
            tokenizer=tokenizer,
            max_seq_length=2048,
            min_seq_length=1,
            add_bos=False,
            add_eos=True,
            add_sep=False,
            sep_id=49704,
            max_num_samples=None, #??
            seed=1234,
            context_key="input",
            label_key="output",
            separate_prompt_and_response_with_newline=True,
            answer_only_loss=True,
            truncation_field="context", #default is answer
            pad_to_max_length=False,  # (@adithyare) allows for much faster training especially in PEFT settings.
            index_mapping_dir=None,
            prompt_template="{input} {output}",
            virtual_tokens=0,
            tokens_to_generate=0
        )

    # Only sending a single prompt for now so taking the first element and
    # ignoring extra fields related to taskname and answer IDs which aren't applicable.
    _, input_ids, _ = dataset[0]
    return input_ids


def convert_prompt(
        use_nemo: bool,
        ptuning_mode: bool,
        prompt: str,
        tokenizer: Any,
        template: dict = None,
        pseudo_tokens: List = None,
        taskname: str = None,
        config: dict = None
    ) -> List:
    """
    Convert the prompt to a list of tokenized IDs using either the NeMo helper
    modules or the standalone method.

    Args:
        prompt: A string of the requested prompt.
        template: A dictionary of the converted prompt template from the model config.
        tokenizer: The requested tokenizer.
        pseudo_tokens: A list of pseudo tokens to replace the virtual prompt placeholder with.
        taskname: The taskname to generate the prompt for.
        config: The config file included with the model.

    Returns:
        Returns a list of IDs from the tokenized prompt.
    """
    if use_nemo:
        input_ids = convert_nemo_prompt(
            ptuning_mode,
            prompt,
            template,
            tokenizer,
            pseudo_tokens,
            taskname,
            config
        )
    else:
        input_ids = convert_hf_prompt(
            ptuning_mode,
            prompt,
            template,
            tokenizer,
            pseudo_tokens,
            taskname
        )
    return input_ids
