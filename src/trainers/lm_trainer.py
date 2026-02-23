import torch

from models.llama import LlamaForCausalLM
from models.custom_llama import CustomLlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class LMTrainer(BaseTrainer):

    model: LlamaForCausalLM | CustomLlamaForCausalLM


    def forward(self, input_ids):

        pad_token_id = self.model.config.pad_token_id

        inputs_for_model = torch.where(
            input_ids != pad_token_id,
            input_ids,
            torch.zeros_like(input_ids)
        )

        logits, _ = self.model(
            input_ids=inputs_for_model,
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

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "atom_count": (input_ids != pad_token_id).long().sum(),
            "logit_nan": (~torch.isfinite(logits)).any().long(),
        }
    