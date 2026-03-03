import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import constants
if constants.XLA_AVAILABLE:
    from torchprime.torch_xla_models import offloading

from torchprime.layers.sequential import HomogeneousSequential

import math
from omegaconf import DictConfig

from models.llama import LlamaForCausalLM, LlamaRMSNorm, LlamaMLP


class HeadLayer(nn.Module):

    def __init__(
        self,
        config: DictConfig
    ):
        super().__init__()

        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        self.mlp = LlamaMLP(config)
    

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        if constants.XLA_AVAILABLE:
            hidden_states = offloading.offload_name(hidden_states, "head_input")

        x = self.norm(hidden_states)
        x = self.mlp(x)

        return hidden_states + x


class NucHead(nn.Module):

    def __init__(
        self,
        config: DictConfig
    ):
        """
        A head for the energy-based model, that predicts whether
        a sample is real or sampled from the model.

        """
        super().__init__()


        self.input_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        self.embed_samples = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = HomogeneousSequential(
            *[
                HeadLayer(config)
                for layer_idx in range(config.num_head_layers)
            ]
        )

        self.out_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.output_proj = nn.Linear(config.hidden_size, 1, bias=False)

    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        sample_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size] hidden states for each token
            sample_ids: [batch_size, sequence_length] or [num_samples, batch_size, sequence_length] the token ids for each token, used to get the sample embeddings
        """
        while sample_ids.dim() > hidden_states.dim() - 1:
            hidden_states = hidden_states.unsqueeze(0)

        hidden_states = (
            self.input_norm(hidden_states) +
            self.embed_samples(sample_ids)
        )

        hidden_states = self.layers(hidden_states)

        hidden_states = self.out_norm(hidden_states)
        pred = self.output_proj(hidden_states).squeeze(-1)

        return pred
    

class NucModel(LlamaForCausalLM):
    
    def __init__(self, config: DictConfig):
        LlamaForCausalLM.__init__(self, config)

        self.head = NucHead(config)

        self._init_head_weights(self.head)
        self.head.output_proj.weight.data.mul_(self.config.head_output_proj_init_scale)

    
    def _init_head_weights(self, module: nn.Module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1/math.sqrt(module.in_features))
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)


    def get_trainable_parameters(self):

        params = {
            'head': set(self.head.parameters()),
        }

        params['main'] = []
        for param in self.parameters():

            if param not in params['head']:
                params['main'].append(param)
            else:
                print("NOT adding parameter to main:", param.shape, flush=True)

        return params

        
    def energy(
        self,
        hidden_states: torch.FloatTensor,
        sample_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Get the energy for a batch of samples.

        Args:
            hidden_states: [batch_size, sequence_length, hidden_size] hidden states for each token
            sample_ids: [batch_size, sequence_length] or [num_samples, batch_size, sequence_length] the token ids for each token, used to get the sample embeddings
        Returns:
            energy: [batch_size, sequence_length] the energy for each token in the sequence
        """
        return self.head(hidden_states, sample_ids)