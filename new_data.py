import torch

import io
import numpy as np


def load_any_array(x) -> torch.Tensor:

    if isinstance(x, list):
        return torch.tensor(x)
    
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    
    elif isinstance(x, torch.Tensor):
        return x
    
    raise ValueError(f"Unsupported type {type(x)} for load_any_array")


def load_byte_array(
    data: bytes
) -> torch.LongTensor:
    """ Convert the data from a byte stream to a tensor.
        - see npy_loads() in https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py
    """

    # depending on datasets? versions, sometimes functions that call this will already be in a list?
    try:
        stream = io.BytesIO(data)
        out = np.lib.format.read_array(stream)
    except:
        out = np.array(data)

    return out


def get_hf_files(
    hf_id: str,
    name: str,
    branch: str = "main",
):
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     - see data_prep.token_wds for repo layout
     
    Args:
        name (str): name of the repo to load

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    data_files = {}
    for split in ["train", "val", "test"]:

        data_files[split] = f"https://huggingface.co/datasets/{hf_id}/{name}/resolve/{branch}/{split}/*"
    
    return data_files