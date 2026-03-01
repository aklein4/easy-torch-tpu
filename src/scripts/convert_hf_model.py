import torch

import argparse
import os

import huggingface_hub as hf
from transformers import AutoModelForCausalLM


TMP_FILE = "tmp_model.pt"


def main(args):
    
    model = AutoModelForCausalLM.from_pretrained(args.in_url, trust_remote_code=True)
    torch.save(
        model.state_dict(),
        TMP_FILE,
    )

    hf.create_repo(
        args.out_url,
        private=False,
        exist_ok=True,
        repo_type="model",
    )

    api = hf.HfApi()
    api.upload_file(
        path_or_fileobj=TMP_FILE,
        path_in_repo=f"{0:012d}/model.pt",
        repo_id=args.out_url,
        repo_type="model",
    )

    os.remove(TMP_FILE)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert HuggingFace model to checkpoint format compatible with easy-torch-tpu.")
    parser.add_argument("--in_url", type=str, required=True, help="HuggingFace model URL to convert.")
    parser.add_argument("--out_url", type=str, required=True, help="HuggingFace model URL to save converted model to.")
    
    args = parser.parse_args()

    main(args)
