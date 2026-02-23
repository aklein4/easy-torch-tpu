import torch

import io
import numpy as np


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
    url: str,
    branch: str = "main",
    splits: list = ["train", "val", "test"]
):
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     
    Args:
        url (str): dataset url on huggingface hub
        branch (str, optional): branch to load from. Defaults to "main".
        splits (list, optional): list of splits to get urls for. Defaults to ["train", "val", "test"].

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    data_files = {}
    for split in splits:

        data_files[split] = f"https://huggingface.co/datasets/{url}/resolve/{branch}/{split}/*"
    
    return data_files
