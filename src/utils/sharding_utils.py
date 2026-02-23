import torch

import utils.constants as constants
if constants.XLA_AVAILABLE:

    import torch_xla.distributed.spmd as xs


def batch_shard_spec(x):
    assert x.dim() >= 1, "Input tensor must have at least 1 dimension to shard on batch."
    return (("data", "fsdp"),) + (None,) * (x.dim() - 1)


def shard_no_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
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


def shard_with_gradients(
    x: torch.Tensor,
    mesh = None,
    spec = None,
) -> torch.Tensor:
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
    if constants.XLA_AVAILABLE:
        return shard_with_gradients(x, mesh, spec)
    else:
        return x
