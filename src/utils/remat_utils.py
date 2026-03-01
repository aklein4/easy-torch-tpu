import torch

import omegaconf

import logging

from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
    add_activation_checkpointing_and_scan,
    add_optimization_barriers,
)

logger = logging.getLogger(__name__)


def advanced_remat(
    model: torch.nn.Module,
    config: omegaconf.DictConfig
) -> torch.nn.Module:
    """
    Apply an advanced rematerialization strategy to the model based on the config.

    This enables multiple submodules to be compiled with a scan (which allows for tensor offloading) and
    nested rematerialization strategies.

    If config.model.remat.advanced is not set, applies remat using the default torchprime config style.
    Otherwise, config.model.remat.advanced should follow this format:
    ```
     -  name: module_name_1
        settings:
            activation_checkpoint_layers:
            - ModuleType1
            - ModuleType2
            ...
            activation_barrier_layers: # these should probably be the same as the activation_checkpoint_layers
            - ModuleType1
            - ModuleType2
            ...

     -  name: module_name_2
        settings:
            activation_checkpoint_layers:
            - ModuleTypeA
            - ModuleTypeB
            ...
            optimization_barrier_layers:
            - ModuleTypeA
            - ModuleTypeB
            ...
            scan_layers: layers_module_name
            offload_tensors:
            - layer_input # as defined by offloading.offload_name in the model
    ```
    
    It is recommended for higher-level submodules to come first in the list.

    Args:
        model (torch.nn.Module): The model to apply rematerialization to.
        config (omegaconf.DictConfig): The configuration containing the rematerialization settings.
    Returns:
        torch.nn.Module: The model with rematerialization applied according to config.model.remat.
    """
    
    # use default torchprime config style
    if not hasattr(config.model.remat, 'advanced'):
        model = add_activation_checkpointing_and_scan(model, config.model.remat)
        model = add_optimization_barriers(model, config.model.remat)

        return model

    advanced_configs = config.model.remat.advanced

    for c in advanced_configs:
        name = c.name
        remat_config = c.settings

        logger.info("Applying advanced rematerialization to: %s", name)

        if name == "self":
            model = add_activation_checkpointing_and_scan(model, remat_config)
            model = add_optimization_barriers(model, remat_config)

        else:
            mod = model.get_submodule(name)

            mod = add_activation_checkpointing_and_scan(mod, remat_config)
            mod = add_optimization_barriers(mod, remat_config)

            model.set_submodule(name, mod)        

    return model