#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Huggingface token
    $2: wandb token

Example:
    . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
'

# update path(?)
export PATH="/home/$USER/.local/bin:$PATH"

# # upgrade and update pip
pip install --upgrade pip
pip install --upgrade setuptools

# install torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# install torch_xla
pip install torch_xla[tpu]==2.8.0
# pip install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

# install extras
pip install setuptools==67.7.2
pip install transformers==4.52.1 datasets==4.0.0 hydra-core==1.3.0 optax==0.2.4 wandb gitpython

# login to huggingface
hf auth login --token $1

# login to wandb
python -m wandb login $2
