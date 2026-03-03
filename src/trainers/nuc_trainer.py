import torch

from models.nuc import NucModel
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn, lm_acc_fn


class NucTrainer(BaseTrainer):

    model: NucModel


    def nuc_loss(
        self,
        input_energy: torch.FloatTensor,
        sample_energy: torch.FloatTensor,
        pad_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        inut_loss = -input_energy.mean()
        sample_loss = sample_energy.exp().mean(0)

        nuc_loss = inut_loss + sample_loss

        nuc_loss = torch.where(
            pad_mask,
            nuc_loss,
            torch.zeros_like(nuc_loss)
        )
        nuc_loss = nuc_loss.sum() / pad_mask.long().sum().float()

        return inut_loss + sample_loss

    
    def nuc_gain(
        self,
        input_energy: torch.FloatTensor,
        sample_energy: torch.FloatTensor,
        pad_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        z = torch.logsumexp(sample_energy, dim=0)

        gain = input_energy - z
        gain = torch.where(
            pad_mask,
            gain,
            torch.zeros_like(gain)
        )
        gain = gain.sum() / pad_mask.long().sum().float()

        return gain
    

    def nuc_acc(
        self,
        input_energy: torch.FloatTensor,
        sample_energy: torch.FloatTensor,
        pad_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        correct = (input_energy[None] > sample_energy).float().mean(0)
        correct = torch.where(
            pad_mask,
            correct,
            torch.zeros_like(correct)
        )
        acc = correct.sum() / pad_mask.long().sum().float()

        return acc
        

    def forward(self, input_ids):

        # this hack gets around some models not having a pad token id in their embeddings
        pad_token_id = self.model.config.pad_token_id
        inputs_for_model = torch.where(
            input_ids != pad_token_id,
            input_ids,
            torch.zeros_like(input_ids)
        )
        pad_mask = (input_ids != pad_token_id)

        logits, _, hidden_states = self.model(
            input_ids=inputs_for_model,
            shift_states=True,
            return_states=True,
        )
        
        sample_ids = self.model.sample(logits, self.config.trainer.num_nuc_samples)
        
        all_ids = torch.cat([inputs_for_model[None, :, 1:], sample_ids], dim=0)
        energy = self.model.energy(
            hidden_states[:, :-1],
            all_ids,
        )
        input_energy, sample_energy = energy[0], energy[1:]

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

        nuc_loss = self.nuc_loss(
            input_energy,
            sample_energy,
            pad_mask[1:],
        )
        nuc_gain = self.nuc_gain(
            input_energy,
            sample_energy,
            pad_mask[1:],
        )
        nuc_acc = self.nuc_acc(
            input_energy,
            sample_energy,
            pad_mask[1:],
        )

        total_lm_loss = lm_loss - nuc_gain

        loss = lm_loss + self.config.trainer.nuc_loss_weight * nuc_loss

        return loss, {
            "lm_loss": lm_loss,
            "lm_acc": lm_acc,
            "nuc_loss": nuc_loss,
            "nuc_gain": nuc_gain,
            "nuc_acc": nuc_acc,
            "total_lm_loss": total_lm_loss,
            "atom_count": pad_mask.long().sum(),
        }
    