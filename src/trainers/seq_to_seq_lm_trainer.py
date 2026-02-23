import torch

from models.custom_llama import CustomLlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn, IGNORE_INDEX


class SeqToSeqLMTrainer(BaseTrainer):

    model: CustomLlamaForCausalLM


    def forward(self, input_ids, output_ids):
        pad_token_id = self.model.config.pad_token_id

        all_ids = torch.cat([input_ids, output_ids], dim=1)
        
        elementwise_pad_mask = (all_ids != pad_token_id)
        ids_for_model = torch.where(
            elementwise_pad_mask,
            all_ids,
            torch.zeros_like(all_ids)
        )

        logits, _ = self.model(
            input_ids=ids_for_model,
            shift_states=True,
            elementwise_pad_mask=elementwise_pad_mask
        )

        labels = torch.cat(
            [
                torch.full_like(input_ids, IGNORE_INDEX),
                torch.where(output_ids != pad_token_id, output_ids, torch.full_like(output_ids, IGNORE_INDEX)),
            ],
            dim=1
        )

        lm_loss = lm_loss_fn(
            logits,
            labels,
            ignore_index=IGNORE_INDEX,
            shift_logits=False,
            shift_labels=True,
        )
        lm_acc = lm_acc_fn(
            logits,
            labels,
            ignore_index=IGNORE_INDEX,
            shift_logits=False,
            shift_labels=True,
        )

        loss = lm_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "atom_count": (output_ids != pad_token_id).long().sum(),
            "logit_nan": (~torch.isfinite(logits)).any().long(),
        }
    