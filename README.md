<div align="center">

# easy-torch-tpu

#### Flexible and Scalable Training with PyTorch on Cloud TPU

</div>
<br /><br />

This repo is based on [torchprime](https://github.com/AI-Hypercomputer/torchprime) and integrated with [Weights & Biases](https://wandb.ai) and [Hugging Face](https://huggingface.co). It is designed to be a flexible framework for training custom research-scale models on Google Cloud TPUs using [PyTorch/XLA](https://github.com/pytorch/xla).

This framework prioritizes:
- Flexibility
- Customizability
- Simplicity
- Ease-of-use
- Research-scale models (1b-10b parameters)

## Features

Without touching the base framework, easy-torch-tpu allows you to:

1. Define custom train step functions (optionally including parameter update logic).
2. Implement new nn.Module-based models.
3. Create optimizers with custom step logic (with auxiliary metric logging).
4. Use custom dataloaders (based on collate functions).
5. Define custom recursive module scanning and activation checkpointing.
6. Define custom activation and parameter sharding configs (with FSDP).
7. Save and load checkpoints with Hugging Face
8. Log training metrics to Weights & Biases

and more...

## Installation

1. Create a single-slice TPU VM with version `tpu-ubuntu2204-base`

2. Clone repo onto all VM devices (see [cli-commands documentation](./docs/cli-commands.md)):
```
git clone https://github.com/aklein4/easy-torch-tpu
```

3. Run the installation script on all VM devices (see [tpu_setup.sh](./tpu_setup.sh) for more info):
```
cd ~/easy-torch-tpu && . tpu_setup.sh <HF_ID> <HF_TOKEN> <WANDB_TOKEN>
```

## Getting Started

The [docs](./docs) folder contains useful information about configuration, training, and customization.

## Contributing

If you find a problem, have a suggestion, or want to contribute, open a GitHub issue.

## Acknowledgements

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
