import torch


class SeqToSeqCollator:

    def __init__(
        self,
        input_length: int,
        output_length: int,
        pad_token_id: int,
    ):
        """
        Collator for sequence-to-sequence tasks.

        Args:
            input_length (int): The maximum length of the input sequences.
            output_length (int): The maximum length of the output sequences.
            pad_token_id (int): The token ID used for padding.
        """

        self.input_length = input_length
        self.output_length = output_length

        self.pad_token_id = pad_token_id

    
    def __call__(
        self,
        batch,
    ):
        return {
            "input_ids": handle_ids(
                batch,
                "input_ids",
                self.input_length,
                self.pad_token_id,
            ),
            "output_ids": handle_ids(
                batch,
                "output_ids",
                self.output_length,
                self.pad_token_id,
            )
        }


def handle_ids(batch, key, sequence_length, pad_token_id):

    input_ids = []
    for x in batch:

        input_ids.append(
            torch.tensor(x[key]).long().flatten()
        )

    # pad to length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id,   
    )
    input_ids = input_ids[:, :sequence_length]
    
    # pad to sequence length
    pad = torch.full(
        (input_ids.shape[0], sequence_length - input_ids.shape[1]),
        pad_token_id,
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

    return input_ids
