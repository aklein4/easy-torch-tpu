import torch

import utils.constants as constants
if constants.XLA_AVAILABLE:
    import torch_xla.distributed.spmd as xs

"""
A collection of sharding utility functions that might be useful.
"""


def batch_shard_spec(
    x: torch.Tensor
):  
    """
    A default sharding spec for sharding along the batch (leading) dimension.

    Args:
        x (torch.Tensor): Input tensor to be sharded.
    """
    assert x.dim() >= 1, "Input tensor must have at least 1 dimension to shard on batch."
    return (("data", "fsdp"),) + (None,) * (x.dim() - 1)


def shard_no_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    """
    Mark the input tensor as sharded, without tracking gradients.

    Sharding is marked out-of-place.
    
    Args:
        x (torch.Tensor): Input tensor to be sharded.
        mesh: Optional sharding mesh to use. If None, will use the global mesh.
        spec: Optional sharding spec to use. If None, will use the default batch shard spec.
    Returns:
        torch.Tensor: Tensor with the same data as x, but marked as sharded.
    """

    if not constants.XLA_AVAILABLE:
        raise RuntimeError(
            "XLA is not available, cannot shard tensor."
        )

    if mesh is None:
        mesh = xs.get_global_mesh()
        assert mesh is not None, "No sharding mesh found."

    if spec is None:
        spec = batch_shard_spec(x)

    return xs.mark_sharding(
        x, mesh, spec
    )


def maybe_shard_no_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    """
    Mark the input tensor as sharded, without tracking gradients.

    Sharding is marked out-of-place.
    
    This is a safe version of shard_no_gradients that will be skipped if XLA is not available, instead of raising an error.

    Args:
        x (torch.Tensor): Input tensor to be sharded.
        mesh: Optional sharding mesh to use. If None, will use the global mesh.
        spec: Optional sharding spec to use. If None, will use the default batch shard spec.
    Returns:
        torch.Tensor: Tensor with the same data as x, but maybe marked as sharded.
    """
    if constants.XLA_AVAILABLE:
        return shard_no_gradients(x, mesh, spec)
    else:
        return x


def shard_with_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    """
    Mark the input tensor as sharded, with gradients.

    Sharding is marked out-of-place.
    
    Args:
        x (torch.Tensor): Input tensor to be sharded.
        mesh: Optional sharding mesh to use. If None, will use the global mesh.
        spec: Optional sharding spec to use. If None, will use the default batch shard spec.
    Returns:
        torch.Tensor: Tensor with the same data as x, but marked as sharded.
    """

    if not constants.XLA_AVAILABLE:
        raise RuntimeError(
            "XLA is not available, cannot shard tensor."
        )

    if mesh is None:
        mesh = xs.get_global_mesh()
        assert mesh is not None, "No sharding mesh found."

    if spec is None:
        spec = batch_shard_spec(x)

    return xs.mark_sharding_with_gradients(
        x, mesh, spec
    )


def maybe_shard_with_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
    """
    Mark the input tensor as sharded, with gradients.

    Sharding is marked out-of-place.
    
    This is a safe version of shard_with_gradients that will be skipped if XLA is not available, instead of raising an error.

    Args:
        x (torch.Tensor): Input tensor to be sharded.
        mesh: Optional sharding mesh to use. If None, will use the global mesh.
        spec: Optional sharding spec to use. If None, will use the default batch shard spec.
    Returns:
        torch.Tensor: Tensor with the same data as x, but maybe marked as sharded.
    """
    if constants.XLA_AVAILABLE:
        return shard_with_gradients(x, mesh, spec)
    else:
        return x
