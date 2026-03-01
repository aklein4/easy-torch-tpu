import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A collection of PyTorch utility functions that might be useful.
"""


def scale_gradient(
    x: torch.Tensor,
    scale: torch.Tensor | float | dict
) -> torch.Tensor:
    """
    Scales the gradient flowing through x by the given scale factor.

    If scale is a dict, it should have a "value" key containing the actual scale factor once the backward pass is executed.
    However, it can be empty during the forward pass, which allows you to set the scale factor after the forward pass is complete.
    
    Args:
        x (torch.Tensor): Input tensor.
        scale (torch.Tensor | float | dict): Scale factor for the gradient.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient scaled by scale during backpropagation.
    """
    return _ScaleGradient.apply(x, scale)

class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        if isinstance(scale, torch.Tensor):
            ctx.save_for_backward(scale)
        else:
            ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):

        if hasattr(ctx, "scale"):
            scale = ctx.scale
        else:
            scale, = ctx.saved_tensors

        if isinstance(scale, dict):
            scale = scale["value"]

        return grad_output * scale, None



def attach_gradient(
    real: torch.Tensor,
    ghost: torch.Tensor,
) -> torch.Tensor:
    """
    Attaches the gradient of `ghost` to `real` during backpropagation, so that
    `ghost` will also receive the same gradient as `real` during backpropagation.

    Args:
        real (torch.Tensor): The tensor whose value is used in the forward pass.
        ghost (torch.Tensor): The tensor which also recieves the gradient during backpropagation.
    Returns:
        torch.Tensor: Tensor with the same data as real, but with the gradient of ghost attached during backpropagation.
    """
    return _AttachGradient.apply(real, ghost)

class _AttachGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, real, ghost):
        return real.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def print_gradient(
    x: torch.Tensor,
    name: str | None = None,
) -> torch.Tensor:
    """
    Print the gradient flowing through x during backpropagation.
    
    If name is provided, it will be printed before the gradient for easier identification.
    
    Args:
        x (torch.Tensor): Input tensor.
        name (str | None): Optional name to identify the gradient.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient printed during backpropagation.
    """
    return _PrintGradient.apply(x, name)

class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):

        if ctx.name is not None:
            print(f"Gradient of {ctx.name}:", flush=True)
        else:
            print("Gradient:", flush=True)
        print(grad_output, flush=True)
        
        return grad_output, None


def transform_gradient(
    x: torch.Tensor,
    fn: callable,
    fn_kwargs: dict={},
) -> torch.Tensor:
    """
    Applies a transformation function to the gradient flowing through x.

    fn should have the signature: fn(x, grad_output, **fn_kwargs) -> transformed_grad_output
    
    Args:
        x (torch.Tensor): Input tensor.
        fn (callable): Function to transform the gradient.
        fn_kwargs (dict): Additional keyword arguments for the function.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient transformed by fn during backpropagation.
    """
    return _TransformGradient.apply(x, fn, fn_kwargs)

class _TransformGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, fn, fn_kwargs):
        ctx.save_for_backward(x)
        ctx.fn = fn
        ctx.fn_kwargs = fn_kwargs
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]

        grad_output = ctx.fn(
            x, grad_output, **ctx.fn_kwargs
        )

        return grad_output, None, None


def unsqueeze_to_batch(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add leading dimensions to x (out-of-place) until it has the same number of dimensions as target.

    Args:
        x (torch.Tensor): Input tensor to be unsqueezed.
        target (torch.Tensor): Target tensor whose number of dimensions we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with leading dimensions added.
    """

    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add leading dimensions to x (out-of-place) until it has the same number of dimensions as target.
    Then expand those leading dimensions to match the corresponding dimensions of target.

    Existing dimensions of x are not changed.

    Args:
        x (torch.Tensor): Input tensor to be expanded.
        target (torch.Tensor): Target tensor whose number of dimensions and leading dimension sizes we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with leading dimensions added and expanded.
    """
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x


def unsqueeze_to_channel(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add trailing dimensions to x (out-of-place) until it has the same number of dimensions as target.

    Args:
        x (torch.Tensor): Input tensor to be unsqueezed.
        target (torch.Tensor): Target tensor whose number of dimensions we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with trailing dimensions added.
    """

    while x.dim() < target.dim():
        x = x[..., None]

    return x


def expand_to_channel(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add trailing dimensions to x (out-of-place) until it has the same number of dimensions as target.
    Then expand those trailing dimensions to match the corresponding dimensions of target.

    Existing dimensions of x are not changed.

    Args:
        x (torch.Tensor): Input tensor to be expanded.
        target (torch.Tensor): Target tensor whose number of dimensions and trailing dimension sizes we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with trailing dimensions added and expanded.
    """
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[..., None]
        num_unsqueeze += 1

    x = x.expand(
        *(list(og_shape) + [target.shape[i] for i in range(num_unsqueeze)])
    )

    return x


def safe_copy_state(
    src: nn.Module,
    dst: nn.Module,
    strict: bool = True,
) -> None:
    """
    Copy the state dict from src to dst, safely cloning and detaching every tensor.
    
    dst is modified in-place, and nothing is returned.

    Args:
        src (nn.Module): Source module to copy state from.
        dst (nn.Module): Destination module to copy state to.
        strict (bool): Whether to strictly enforce that the keys in src and dst match.
    """

    state = {
        k: v.clone().detach() for k, v in src.state_dict().items()
    }

    dst.load_state_dict(state, strict=strict)
