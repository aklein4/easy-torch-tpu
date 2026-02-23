
import utils.constants as constants
if constants.XLA_AVAILABLE:

    import torch_xla.distributed.spmd as xs

    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding


def master_print(*args, flush=True, **kwargs):
    if not constants.XLA_AVAILABLE or constants.PROCESS_IS_MAIN():
        print(*args, flush=flush, **kwargs)


def print_sharding_info(tensor, name=None):
    
    if constants.XLA_AVAILABLE:
        xt = xs.wrap_if_sharded(tensor)

        master_print(f" ===== Sharding info for {name if name else 'tensor'} ===== ")
        master_print(f" - Shape: {xt.shape}")
        if isinstance(xt, xs.XLAShardedTensor):
            master_print(f" - Sharding spec: {xt.sharding_spec}\n")
        else:
            master_print(" - Not sharded\n")

    else:
        raise RuntimeError("Cannot print sharding info. XLA is not available.")


def visualize_sharding_info(tensor, name=None):

    if constants.XLA_AVAILABLE:
        xt = xs.wrap_if_sharded(tensor)

        master_print(f" ===== Sharding info for {name if name else 'tensor'} ===== ")
        master_print(f" - Shape: {xt.shape}")
        if isinstance(xt, xs.XLAShardedTensor):
            table = visualize_tensor_sharding(xt)
            master_print(table)
            master_print("")
        else:
            master_print(" - Not sharded\n")

    else:
        raise RuntimeError("Cannot visualize sharding info. XLA is not available.")
    