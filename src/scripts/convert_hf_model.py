import torch

import os

import huggingface_hub as hf
from transformers import AutoModelForCausalLM


IN_URL = 'HuggingFaceTB/SmolLM2-1.7B'
OUT_URL = 'aklein4/SmolLM2-1.7B-TPU'

TMP_FILE = "tmp_model.pt"

def main():
    
    model = AutoModelForCausalLM.from_pretrained(IN_URL)
    torch.save(
        model.state_dict(),
        TMP_FILE,
    )

    hf.create_repo(
        OUT_URL,
        private=False,
        exist_ok=True,
        repo_type="model",
    )

    api = hf.HfApi()
    api.upload_file(
        path_or_fileobj=TMP_FILE,
        path_in_repo=f"{0:012d}/model.pt",
        repo_id=OUT_URL,
        repo_type="model",
    )

    os.remove(TMP_FILE)


if __name__ == '__main__':
    main()
