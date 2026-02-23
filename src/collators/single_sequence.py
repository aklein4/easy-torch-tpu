import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SingleSequenceCollator:

    def __init__(
        self,
        sequence_length: int,
        pad_token_id: int,
        vocab_size: int,
    ):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    
    def __call__(
        self,
        batch,
    ):

        input_ids = []
        for x in batch:

            in_ids = torch.tensor(x["input_ids"]).long().flatten()
            out_ids = torch.tensor(x["output_ids"]).long().flatten()

            input_ids.append(torch.cat([in_ids, out_ids], dim=0))

        # pad to length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,   
        )
        input_ids = input_ids[:, :self.sequence_length]
        
        # pad to sequence length
        pad = torch.full(
            (input_ids.shape[0], self.sequence_length - input_ids.shape[1]),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        input_ids = torch.cat(
            [
                input_ids,
                pad
            ],
            dim=1
        )

        input_ids = torch.clip(input_ids, 0, self.vocab_size - 1)

        return input_ids
