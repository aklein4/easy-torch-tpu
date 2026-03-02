import torch

import math

from models.vae import VAEModel
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class VAETrainer(BaseTrainer):

    model: VAEModel


    def forward(self, input_ids):

        # this hack gets around some models not having a pad token id in their embeddings
        pad_token_id = self.model.config.pad_token_id
        inputs_for_model = torch.where(
            input_ids != pad_token_id,
            input_ids,
            torch.zeros_like(input_ids)
        )

        mu, sigma = self.model.encode(inputs_for_model)

        z = mu + sigma * torch.randn_like(sigma)

        logits = self.model.decode(
            input_ids=inputs_for_model,
            z=z,
            shift_states=True,
        )

        lm_loss = lm_loss_fn(
            logits,
            input_ids,
            ignore_index=pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            input_ids,
            ignore_index=pad_token_id,
            shift_logits=False,
            shift_labels=True,
        )

        total_kl = torch.sum(
            torch.log(sigma) + (sigma**2 + mu**2) / 2 - 0.5
        )
        num_tokens = (input_ids != pad_token_id).long().sum()
        kl_per_token = total_kl / num_tokens.float()

        loss = (
            lm_loss +
            self.config.trainer.kl_weight * kl_per_token
        )

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "kl_per_token": kl_per_token,
            "atom_count": num_tokens,
        }
    