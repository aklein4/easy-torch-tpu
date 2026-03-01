#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command to set the environment variables in the current shell.

Arguments:
    $1: Hugging Face ID (ex. aklein4)
    $2: Hugging Face token
    $3: wandb token

Example:
    . tpu_setup.sh <HF_ID> <HF_TOKEN> <WANDB_TOKEN>
'

# update path(?)
export PATH="/home/$USER/.local/bin:$PATH"

# # upgrade and update pip
pip install --upgrade pip
pip install --upgrade setuptools

# install torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_xla[tpu]==2.8.0

# install extras
pip install setuptools==67.7.2
pip install -r tpu_requirements.txt

# login to huggingface
hf auth login --token $2

# login to wandb
python -m wandb login $3

# create a .env file to store the Hugging Face ID
echo "HF_ID=$1" > .env
