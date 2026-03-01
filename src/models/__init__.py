""" Models """

import torch

import omegaconf
import os
import huggingface_hub as hf
import shutil

from utils.import_utils import import_model
from utils import constants


def load_checkpoint(
    url: str,
    step: int,
    attention_kernel: str = "other", # uses non-kernel attention by default
    strict: bool = True,
    ignore_cache: bool = False,
    remove_folder: bool = False,
    model_type: str = None,
    skip_state_dict: bool = False,
) -> torch.nn.Module:
    """
    Loads a model checkpoint from Hugging Face Hub or local folder.

    Args:
        url (str): The URL or local path of the checkpoint to load.
        step (int): The training step corresponding to the checkpoint to load.
        attention_kernel (str): The attention kernel used in the model. Default is "other" for non-kernel attention.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict(). Default is True.
        ignore_cache (bool): Whether to ignore the local cache and redownload the checkpoint. Default is False.
        remove_folder (bool): Whether to remove the downloaded checkpoint folder after loading. Default is False.
        model_type (str): The type of the model to load. If None, it will be inferred from the checkpoint config. Default is None.
        skip_state_dict (bool): Whether to skip loading the state dict and only initialize the model architecture. Default is False.
    """
    
    name = url.replace("/", "--")
    local_path = os.path.join(constants.CHECKPOINTS_PATH, name)
    
    subfolder = f"{step:012d}"
    subfolder_path = os.path.join(local_path, subfolder)

    if ignore_cache:
        shutil.rmtree(subfolder_path, ignore_errors=True)
        
    if not os.path.exists(subfolder_path):
        hf.snapshot_download(
            repo_id=url,
            allow_patterns=[subfolder + "/*"],
            local_dir=local_path,
        )

    # load the model
    config_path = os.path.join(subfolder_path, "config.json")
    config = omegaconf.OmegaConf.load(config_path)
    config.attention_kernel = attention_kernel

    if model_type is None:
        model_type = config.type
    model = import_model(model_type)(config)

    # load the weights
    if not skip_state_dict:
        state_path = os.path.join(subfolder_path, "model.pt")
        state_dict = torch.load(state_path, map_location="cpu")

        # remove and xla specific keys
        cleaned_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(cleaned_state_dict, strict=strict)

    if remove_folder:
        shutil.rmtree(subfolder_path, ignore_errors=True)

    return model


def load_checkpoint_state(
    model: torch.nn.Module,
    url: str,
    step: int,
    strict: bool = True,
    ignore_cache: bool = False,
    remove_folder: bool = False,
):
    """
    Loads a checkpoint state dict from Hugging Face Hub or local folder into a given model.

    Args:
        model (torch.nn.Module): The model into which the checkpoint state dict will be loaded.
        url (str): The URL or local path of the checkpoint to load.
        step (int): The training step corresponding to the checkpoint to load.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict(). Default is True.
        ignore_cache (bool): Whether to ignore the local cache and redownload the checkpoint. Default is False.
        remove_folder (bool): Whether to remove the downloaded checkpoint folder after loading. Default is False.
    """
    
    name = url.replace("/", "--")
    local_path = os.path.join(constants.CHECKPOINTS_PATH, name)
    
    subfolder = f"{step:012d}"
    subfolder_path = os.path.join(local_path, subfolder)

    if ignore_cache:
        shutil.rmtree(subfolder_path, ignore_errors=True)
        
    if not os.path.exists(subfolder_path):
        hf.snapshot_download(
            repo_id=url,
            allow_patterns=[subfolder + "/*"],
            local_dir=local_path,
        )

    # load the weights
    state_path = os.path.join(subfolder_path, "model.pt")
    state_dict = torch.load(state_path, map_location="cpu")

    # remove and xla specific keys
    cleaned_state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict, strict=strict)

    if remove_folder:
        shutil.rmtree(subfolder_path, ignore_errors=True)

    return model
