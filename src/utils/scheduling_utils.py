import torch


def linear_warmup(
    step: torch.Tensor | int,
    warmup_steps: int,
) -> torch.Tensor | float:
    """ Linear warmup function.

    Args:
        step (torch.Tensor | int): current step
        warmup_steps (int): number of warmup steps

    Returns:
        torch.Tensor | float: value between 0.0 and 1.0 representing progress through warmup
    """

    if isinstance(step, torch.Tensor):
        return torch.clamp(step / warmup_steps, 0.0, 1.0)
    
    return max(0.0, min(1.0, step / warmup_steps))
