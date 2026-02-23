import torch


class SimpleCollator:

    def __init__(
        self,
    ):
        """
        A simple collator for pre-tokenized data with equal length.
        """
        return

    
    def __call__(
        self,
        batch,
    ):  
        
        input_ids = []
        for x in batch:
            input_ids.append(
                torch.tensor(x["input_ids"]).long().flatten()
            )

        input_ids = torch.stack(input_ids, dim=0)

        return {
            "input_ids": input_ids
        }
