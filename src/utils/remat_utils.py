import torch
import torch.nn as nn

import logging

from torchprime.torch_xla_models.model_rewriting.rematerialization_utils import (
    add_activation_checkpointing_and_scan,
    add_optimization_barriers,
)

logger = logging.getLogger(__name__)


def advanced_remat(
    model: nn.Module,
    config
) -> nn.Module:
    
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