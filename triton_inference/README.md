# Sending Inference Requests for P-Tuned, LoRA, and SFT Models
This repository contains scripts and documentation to deploy SFT, LoRA, and P-tuned Models (.nemo) using FasterTransformer and Triton Inference Server and send requests.

## Running
**NOTE: SFT and LoRA inference methods are currently configured to only use the Hugging Face tokenizer, and thus do not support using NeMo modules for tokenization. P-tuning has support for Hugging Face and NeMo tokenization.**

To generate a prompt, open the `request.py` file and change the `PROMPT` variable at the top of the script. Then, run the request script and specify the path to the downloaded tuned model like so:

```
pip3 install 'nemo_toolkit[nlp]' torch 'tritonclient[http]' transformers

#GPT inference (Hugging Face tokenizer)
python3 request.py --server <server url> --model-name gpt3_5B

#SFT inference (Hugging Face tokenizer)
python3 request.py --server <server url> --model-name sft_gpt3_5B

#LoRA inference (Hugging Face tokenizer)
python3 request.py --server <server url> --model-name lora_gpt3_5B

#P-tuning inference (Hugging Face tokenizer and NeMo tokenizer)
python3 request.py --server <server url> --ptuned-model <ptuned .nemo file> --taskname squad
python3 request.py --server <server url> --ptuned-model <ptuned .nemo file> --taskname squad --use-nemo

```
