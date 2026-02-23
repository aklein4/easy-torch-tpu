import torch
import torch.nn as nn
import torch.nn.functional as F


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

def scale_gradient(
    x: torch.Tensor,
    scale: torch.Tensor | float | dict
) -> torch.Tensor:
    """
    Scales the gradient flowing through x by the given scale factor.

    Args:
        x (torch.Tensor): Input tensor.
        scale (torch.Tensor | float | dict): Scale factor for the gradient.
    """
    return _ScaleGradient.apply(x, scale)


class _AttachGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, real, ghost):
        return real.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def attach_gradient(
    real: torch.Tensor,
    ghost: torch.Tensor,
) -> torch.Tensor:
    """
    Attaches the gradient of `ghost` to `real` during backpropagation.

    Args:
        real (torch.Tensor): The tensor whose value is used in the forward pass.
        ghost (torch.Tensor): The tensor who also recieves the gradient during backpropagation.
    """
    return _AttachGradient.apply(real, ghost)


class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):

        if ctx.name is not None:
            print(f"Gradient of {ctx.name}:")
        else:
            print("Gradient:")
        print(grad_output)
        
        return grad_output, None

def print_gradient(x, name=None):
    return _PrintGradient.apply(x, name)


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

def transform_gradient(
    x: torch.Tensor,
    fn: callable,
    fn_kwargs: dict={},
) -> torch.Tensor:
    """
    Applies a transformation function to the gradient flowing through x.

    Args:
        x (torch.Tensor): Input tensor.
        fn (callable): Function to transform the gradient.
        fn_kwargs (dict): Additional keyword arguments for the function.
    """
    return _TransformGradient.apply(x, fn, fn_kwargs)


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x


def unsqueeze_to_channel(x, target):
    while x.dim() < target.dim():
        x = x[..., None]

    return x


def expand_to_channel(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[..., None]
        num_unsqueeze += 1

    x = x.expand(
        *(list(og_shape) + [target.shape[i] for i in range(num_unsqueeze)])
    )

    return x


def safe_copy_state(src, dst, strict=True):

    state = {
        k: v.clone().detach() for k, v in src.state_dict().items()
    }

    dst.load_state_dict(state, strict=strict)


def shift(
    x: torch.Tensor,
    n: int,
    dim: int,
    direction: str,
    narrow: bool,
):

    zero_shape = list(x.shape)
    zero_shape[dim] = n
    z = torch.zeros(*zero_shape, device=x.device, dtype=x.dtype)

    if direction == 'right':
        if narrow:
            x = torch.narrow(x, dim, 0, x.shape[dim] - n)
        
        l = [z, x]

    elif direction == 'left':
        if narrow:
            x = torch.narrow(x, dim, n, x.shape[dim] - n)
        
        l = [x, z]

    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    return torch.cat(l, dim=dim)


def gaussian_init(module: nn.Module):

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1/module.in_features**0.5)
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1)


def safe_repeat(
    x: torch.Tensor, n_repeats: int, dim: int=0
) -> torch.Tensor:
    return torch.cat(
        [x] * n_repeats,
        dim=dim
    )
