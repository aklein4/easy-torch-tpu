import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from models.llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer


class VAEEncoderLayer(LlamaDecoderLayer):
    offload_name = "encoder_input"
    is_causal = False

class VAEEncoderModel(LlamaModel):
    layer_type = VAEEncoderLayer


class VAEDecoderLayer(LlamaDecoderLayer):
    offload_name = "decoder_input"
    is_causal = True

class VAEDecoderModel(LlamaModel):
    layer_type = VAEDecoderLayer

class VAEDecoderForCausalLM(LlamaForCausalLM):
    transformer_type = VAEDecoderModel

    def _init_weights(self, module):
        return


class VAEModel(nn.Module):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # save config
        self.latent_size = config.latent_size

        # create the transformer backbones
        self.encoder_model = VAEEncoderModel(config)
        self.decoder_model = VAEDecoderForCausalLM(config)

        # only use a single embedding layer
        self.embed_tokens = self.encoder_model.embed_tokens
        self.encoder_model.embed_tokens = None
        self.decoder_model.model.embed_tokens = None

        # the latent projection layers
        self.encoder_mu_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)
        self.encoder_log_sigma_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)

        self.decoder_latent_proj = nn.Linear(self.latent_size, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    
    def _init_weights(self, module: nn.Module):
        """Initialize weights for Linear and Embedding layers.

        This method initializes the weights of Linear and Embedding layers
        using a normal distribution with mean 0 and standard deviation specified
        by `self.config.initializer_range`. Biases are initialized to zero.

        Args:
            module: The module whose weights need to be initialized.
        """
        std = self.config.initializer_range
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    def encode(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        
        input_embeds = self.embed_tokens(input_ids)

        hidden_states = self.encoder_model(
            input_embeds=input_embeds,
        )

        # take the mean of the hidden states across the sequence dimension
        pooled_hidden_states = hidden_states.mean(dim=-2)

        # project to the latent space
        mu = self.encoder_mu_proj(pooled_hidden_states)
        sigma = F.softplus(self.encoder_log_sigma_proj(pooled_hidden_states))

        return mu, sigma
    

    def decode(
        self,
        input_ids: torch.LongTensor,
        z: torch.FloatTensor,
        shift_states: bool = False,
    ):
        
        inputs_embeds = (
            self.embed_tokens(input_ids) +
            self.decoder_latent_proj(z).unsqueeze(1)
        )

        logits, _ = self.decoder_model(
            inputs_embeds=inputs_embeds,
            shift_states=shift_states,
        )

        return logits
    