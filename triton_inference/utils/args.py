import numpy as np
from argparse import ArgumentParser, Namespace


def arguments() -> Namespace:
    """
    Parse arguments during runtime.
    """
    parser = ArgumentParser()

    # P-tuning specific arguments
    parser.add_argument('--ptuned-model', '-m', type=str, default=None, required=False,
                        help='Specify the path to the tuned model. Can '
                        'either be a *.nemo file or a directory pointing to '
                        'an extracted model.')
    parser.add_argument('--taskname', type=str, default='taskname',
                        help='Specify a custom taskname if necessary. Most '
                        'models will use "taskname" by default.')
    parser.add_argument('--use-nemo', action='store_true',
                        help='Use NeMo modules for tokenizing the prompt. NeMo '
                        'takes longer to load but could be more flexible for '
                        'complex prompts.')

    # Model-specific arguments
    parser.add_argument('--topk', '-k', type=np.uint32, default=0,
                        help='Specify the top-k value for the model.')
    parser.add_argument('--topp', '-p', type=float, default=0.9,
                        help='Specify the top-p value for the model.')
    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                        help='Specify the temperature value for the model.')
    parser.add_argument('--len-penalty', '-l', type=float, default=1.0,
                        help='Specify the length penalty for the model.')
    parser.add_argument('--repetition-penalty', '-r', type=float, default=1.0,
                        help='Specify the repetition penalty for the model.')
    parser.add_argument('--max-output-len', type=int, default=10,
                        help='Specify the maximum desired number of tokens to '
                        'generate for the model.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Specify the random seed to use for the request.')
    parser.add_argument('--beam_width', '-b', type=int, default=1,
                        help='Specify the beam width value for the model.')

    # Server-specific arguments
    parser.add_argument('--server', '-s', type=str, default='localhost:8000',
                        help='Specify the hostname:port where the inference '
                        'server is deployed. Defaults to "localhost:8000".')
    parser.add_argument('--model-name', type=str, default='gpt3_5b',
                        help='Specify the name of the inference model that was '
                        'deployed with Triton Inference Server.')
    return parser.parse_args()
