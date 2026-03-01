import torch

from transformers import AutoTokenizer


class TokenizeCollator:

    def __init__(
        self,
        tokenizer_url: str,
        sequence_length: int,
        text_key: str = "text",
    ):
        """
        A simple collator for tokenizing raw text data with truncation and right-padding to a fixed length.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        if self.tokenizer.pad_token is None:
            # in this case, your model should have pad_token_id set equal to vocab_size
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.sequence_length = sequence_length
        self.text_key = text_key

    
    def __call__(
        self,
        batch,
    ):  
        
        text = [x[self.text_key] for x in batch]

        input_ids = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.sequence_length,
            return_tensors="pt"
        )["input_ids"].long()

        return {
            "input_ids": input_ids
        }
