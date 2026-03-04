from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

import math

from utils.torch_utils import safe_finite


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
        update_clip (:obj:`float`, `optional`, defaults to `None`):
            If a float value is provided, the update is clipped to the range :obj:`[-update_clip, update_clip]`.
        fix_nan (:obj:`bool`, `optional`, defaults to `True`):
            Whether to fix NaN values in gradients and updates by replacing them with zeros.
        state_dtype (:obj:`torch.dtype`, `optional`, defaults to "float32"):
            The data type to use for the optimizer state (e.g., "float32", "bfloat16", "float16").
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        update_clip: float = None,
        fix_nan: bool = True,
        state_dtype: torch.dtype = "float32",
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
            update_clip=update_clip, fix_nan=fix_nan, state_dtype=getattr(torch, state_dtype)
        )
        
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grad_nan = torch.tensor([False], device=self.param_groups[0]['params'][0].device)
        update_nan = torch.tensor([False], device=self.param_groups[0]['params'][0].device)
        param_nan = torch.tensor([False], device=self.param_groups[0]['params'][0].device)
        post_param_nan = torch.tensor([False], device=self.param_groups[0]['params'][0].device)
        for group in self.param_groups:
            for p in group["params"]:
                
                # get the grad
                if p.grad is None:
                    continue
                grad = p.grad

                # handle types
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                # handle nan gradients
                grad_nan = grad_nan | (~torch.isfinite(grad)).any()
                if group["fix_nan"]:
                    grad = safe_finite(grad)

                param_nan = param_nan | (~torch.isfinite(p)).any()

                state = self.state[p]
                state_dtype = group["state_dtype"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad, dtype=state_dtype)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=state_dtype)
                else:
                    state["exp_avg"] = state["exp_avg"].to(grad.device, dtype=state_dtype)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad.device, dtype=state_dtype)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(
                    grad.to(exp_avg.dtype),
                    alpha=1.0 - beta1
                )
                exp_avg_sq.mul_(beta2).add_(
                    grad.pow(2).to(exp_avg_sq.dtype),
                    alpha=1.0 - beta2
                )

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                denom = torch.clamp(
                    exp_avg_sq.to(p.dtype), min=group["eps"]**2
                ).sqrt()
                update = exp_avg.to(p.dtype) / denom
                if group["update_clip"] is not None:
                    update = torch.clamp(update, -group["update_clip"], group["update_clip"])

                update_nan = update_nan | (~torch.isfinite(update)).any()
                if group["fix_nan"]:
                    update = safe_finite(update)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                p.add_(update, alpha=-step_size)

                post_param_nan = post_param_nan | (~torch.isfinite(p)).any()

        return {
            "grad_nan": grad_nan.long(),
            "update_nan": update_nan.long(),
            "param_nan": param_nan.long(),
            "post_param_nan": post_param_nan.long()
        }
    