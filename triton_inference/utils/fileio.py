import io
import os
import tarfile
from omegaconf import OmegaConf
from typing import Tuple


def _read_extracted_model(model_path: str) -> Tuple[str, str]:
    """
    Finds the paths to the model weights and config files from the extracted
    NeMo file that was specified.

    Args:
        model_path: The path to the extraced model file.

    Returns:
        tuple of the paths to the model weights and config files.
    """
    weights_path = os.path.join(model_path, 'model_weights.ckpt')
    mp_weights_path = os.path.join(model_path, 'mp_rank_00/model_weights.ckpt')
    config_path = os.path.join(model_path, 'model_config.yaml')

    if not os.path.isfile(weights_path) and not os.path.isfile(mp_weights_path):
        raise Exception('Unable to find model_weights.ckpt from the extracted '
                        'ptuned model. Ensure there is a valid file from the '
                        'checkpoint.')
    if not os.path.isfile(config_path):
        raise Exception('Unable to find model_config.yaml from the extracted '
                        'ptuned model. Ensure there is a valid file from the '
                        'checkpoint.')
    if os.path.isfile(mp_weights_path):
        weights_path = mp_weights_path
    return weights_path, config_path


def _read_nemo_file(model_path: str) -> Tuple[str, str]:
    """
    Extracts the NeMo file into a temporary directory and pulls the model
    weights and config files.

    Args:
        model_path: The path to the .nemo file to extract.

    Returns:
        tuple of the paths to the model weights and config files.
    """
    with open(model_path, 'rb') as nemo_file:
        ptuned_model_archive = io.BytesIO(nemo_file.read())
        model_tar = tarfile.open(fileobj=ptuned_model_archive)

        # The files can start with "model_" or "./model_". Looping through the
        # names and matching based on filename will capture either format.
        for filename in model_tar.getnames():
            if 'model_weights.ckpt' in filename:
                model_weights = model_tar.extractfile(filename)
            elif 'model_config.yaml' in filename:
                model_config = model_tar.extractfile(filename)

    if not model_weights or not model_config:
        raise Exception('Unable to find model files from ptuned model. Ensure '
                        'the checkpoint has the expected format.')
    return model_weights, model_config


def _is_nemo(model_path: str) -> bool:
    """
    Checks if the specified model is a .nemo file or an extracted version of
    the model.

    Args:
        model_path: The path to the model file to read from.

    Returns:
        bool, whether the model is a .nemo file or not.
    """
    if model_path.endswith('.nemo'):
        return True
    else:
        return False


def read_model(model_path: str) -> Tuple[str, str]:
    """
    Read the specified model path and retrieve the embedded model weights and
    config from the file.

    Args:
        model_path: The path to the model file to read from.

    Returns:
        tuple of the paths to the model weights and config files.
    """
    if _is_nemo(model_path):
        model_weights, model_config = _read_nemo_file(model_path)
    else:
        model_weights, model_config = _read_extracted_model(model_path)
    return model_weights, model_config


def read_config(model_config: str) -> dict:
    """
    Load the model config.

    Args:
        model_config: The path to the model config file to read from.

    Returns:
        Returns the loaded model config file as an OmegaConf object.
    """
    return OmegaConf.load(model_config)
