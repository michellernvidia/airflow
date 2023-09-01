import numpy as np
import tritonclient.http as httpclient
from argparse import Namespace
from tritonclient.utils import np_to_triton_dtype
from typing import Any, List


def fill_input(input_name: str, data: Any) -> httpclient.InferInput:
    """
    Converts an input from a numpy array to a Triton compatible data type.

    Args:
        input_name: The name of the input parameter to send to Triton.
        data: The full data field as a numpy array.

    Returns:
        Returns the converted field as a Triton input.
    """
    infer_input = httpclient.InferInput(
        input_name,
        data.shape,
        np_to_triton_dtype(data.dtype)
    )
    infer_input.set_data_from_numpy(data)
    return infer_input


def generate_inputs(
        args: Namespace,
        input_ids: List,
        ptuning_mode: bool, 
        request_prompt_embedding: Any = None,
        request_prompt_lengths: np.array = None,
        request_prompt_type: np.array = None
    ) -> List:
    """
    Convert all of the model inputs to Triton compatible data types and create
    a list of inputs to send to the deployed model.

    Args:
        args: The default and user-specified arguments.
        input_ids: A list containing the input prompt and prompt template converted to tokenized IDs.
        ptuning_mode: A bool indicator of whether we're serving a ptuned model or not.
        request_prompt_embedding: An array of the embeddings from the prompt table.
        request_prompt_lengths: An array of the length of each prompt.
        request_prompt_type: An array of the list of prompt data types.

    Returns:
        Returns a list of all inputs converted to Triton compatible data types.
    """
    input_start_ids = np.array([input_ids]).astype(np.uint32)
    input_length = np.array([[len(input_start_ids[0])]]).astype(np.uint32)
    output_len = np.ones_like(input_length).astype(np.uint32) * args.max_output_len

    runtime_top_k = np.array([[args.topk]]).astype(np.uint32)
    runtime_top_p = np.array([[args.topp]]).astype(np.float32)
    beam_search_diversity_rate = np.array([[0.0]]).astype(np.float32)
    temperature = np.array([[args.temperature]]).astype(np.float32)
    len_penalty = np.array([[args.len_penalty]]).astype(np.float32)
    repetition_penalty = np.array([[args.repetition_penalty]]).astype(np.float32)
    random_seed = np.array([[args.random_seed]]).astype(np.uint64)
    is_return_log_probs = np.array([[True]]).astype(bool)
    beam_width = np.array([[args.beam_width]]).astype(np.uint32)
    
    start_ids = np.array([[50256]]).astype(np.uint32)
    end_ids = np.array([[50256]]).astype(np.uint32)
    bad_words_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
        np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)
    stop_word_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
        np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)

    inputs = [
        fill_input("input_ids", input_start_ids),
        fill_input("input_lengths", input_length),
        fill_input("request_output_len", output_len),
        fill_input("runtime_top_k", runtime_top_k),
        fill_input("runtime_top_p", runtime_top_p),
        fill_input("beam_search_diversity_rate", beam_search_diversity_rate),
        fill_input("temperature", temperature),
        fill_input("len_penalty", len_penalty),
        fill_input("repetition_penalty", repetition_penalty),
        fill_input("random_seed", random_seed), 
        fill_input("is_return_log_probs", is_return_log_probs),
        fill_input("beam_width", beam_width), 
        fill_input("start_id", start_ids), 
        fill_input("end_id", end_ids), 
        fill_input("bad_words_list", bad_words_list),
        fill_input("stop_words_list", stop_word_list),
    ]

    if ptuning_mode == True:
        inputs.append(fill_input("request_prompt_embedding", request_prompt_embedding))
        inputs.append(fill_input("request_prompt_lengths", request_prompt_lengths))
        inputs.append(fill_input("request_prompt_type", request_prompt_type))

    return inputs


def send_prompt(server: str, model_name: str, request_inputs: List) -> np.ndarray:
    """
    Send the complete prompt and all parameters to the deployed base model
    running on Triton Inference Server.

    Args:
        server: The hostname and port of the server to send the inference request to.
        model_name: The name of the deployed model on Triton Inference Server.
        request_inputs: A list of all inputs to pass to the model via the Triton client.

    Returns:
        Returns the generated response as a numpy array of IDs to be decoded.
    """
    with httpclient.InferenceServerClient(server, ssl=True) as client:
        result = client.infer(model_name, request_inputs)
        output = result.as_numpy('output_ids').squeeze()
        return output
