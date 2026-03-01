import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


IGNORE_INDEX = -100


def lm_loss_fn(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_index: int | None = None,
    shift_logits: bool = True,
    shift_labels: bool = True,
) -> torch.FloatTensor:
    """ Compute language modeling loss.

    Args:
        logits (torch.FloatTensor): model output logits of shape (B, T, V)
        targets (torch.LongTensor): target token ids of shape (B, T)
        ignore_index (int, optional): index to ignore in loss computation. Defaults to -100.
        shift_logits (bool, optional): whether to shift logits for next-token prediction. Defaults to True.
        shift_labels (bool, optional): whether to shift labels for next-token prediction. Defaults to True.

    Returns:
        torch.FloatTensor: computed loss
    """

    if shift_logits:
        logits = logits[:, :-1, :]
    if shift_labels:
        labels = labels[:, 1:]

    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def lm_acc_fn(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    ignore_index: int | None = None,
    shift_logits: bool = True,
    shift_labels: bool = True,
) -> torch.FloatTensor:
    """ Compute language modeling accuracy.

    Args:
        logits (torch.FloatTensor): model output logits of shape (B, T, V)
        targets (torch.LongTensor): target token ids of shape (B, T)
        ignore_index (int, optional): index to ignore in loss computation. Defaults to -100.
        shift_logits (bool, optional): whether to shift logits for next-token prediction. Defaults to True.
        shift_labels (bool, optional): whether to shift labels for next-token prediction. Defaults to True.

    Returns:
        torch.FloatTensor: computed loss
    """

    if shift_logits:
        logits = logits[:, :-1, :]
    if shift_labels:
        labels = labels[:, 1:]

    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    mask = labels != ignore_index

    correct = (logits.argmax(dim=-1) == labels) & mask

    return correct.float().sum() / (1e-6 + mask.float().sum())
