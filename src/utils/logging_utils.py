import torch

import utils.constants as constants
if constants.XLA_AVAILABLE:
    import torch_xla.distributed.spmd as xs


def master_print(
    *args,
    flush: bool=True,
    **kwargs
) -> None:
    """
    Print only if this is the main process or XLA is not available.
    """
    if not constants.XLA_AVAILABLE or constants.PROCESS_IS_MAIN():
        print(*args, flush=flush, **kwargs)


def print_sharding_info(
    x: "torch.Tensor",
    name: str = None,
) -> None:
    """
    Print the sharding information of a tensor. Only prints on the main process of each device.

    Args:
        x (torch.Tensor): The tensor to print sharding info for.
        name (str, optional): An optional name to identify the tensor in the printout.
    """
    
    if constants.XLA_AVAILABLE:
        xt = xs.wrap_if_sharded(x)

        master_print(f" ===== Sharding info for {name if name else 'tensor'} ===== ")
        master_print(f" - Shape: {xt.shape}")
        if isinstance(xt, xs.XLAShardedTensor):
            master_print(f" - Sharding spec: {xt.sharding_spec}\n")
        else:
            master_print(" - Not sharded\n")

    else:
        raise RuntimeError("Cannot print sharding info. XLA is not available.")
