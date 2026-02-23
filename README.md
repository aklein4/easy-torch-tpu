# easy-torch-tpu

![A banner image showing a TPU board with tensors expanding from it.](./assets/readme_header_image.png)

A flexible framework for training custom research-scale models on Google's TPUs using torch-xla.

Based on [torchprime](https://github.com/AI-Hypercomputer/torchprime) and integrated with Weights & Biases and Huggingface.

As an alternative to [torchprime](https://github.com/AI-Hypercomputer/torchprime), this framework prioritizes:
- flexibility
- customizability
- simplicity
- ease-of-use
- research scale training (1b-10b parameters)

*This repo is a work in progress, with plans to improve documentation and create more example implementations*.

## Features

Without changing the base code, easy-torch-tpu allows you to:

1. Define custom train step functions (optionally including optimization logic).
2. Implement novel nn.Module-based models.
3. Create optimizers with custom step logic with auxiliary metric logging.
4. Use custom dataloaders using collate classes.
5. Save and load checkpoints with non-xla compatibility
6. Define custom recursive module scanning and activation checkpointing.
7. Define custom activation and parameter sharding configs (with FSDP).


## Getting Started

### Installation

1. Create a single-slice TPU VM with version `tpu-ubuntu2204-base`
2. Run command on all devices (see below): `git clone https://github.com/aklein4/easy-torch-tpu`
3. Run the installation script on all devices: `cd ~/easy-torch-tpu && . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>`


### Running Commands

Run a command on all devices within a VM/node:

```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='<COMMAND>'
```

Run a command (such as a training run) in the background that continues running after the terminal closes:
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='nohup <COMMAND> > command.log 2>&1 &'
```
Monitor a background command:
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='tail -f command.log'
```

Run another torch-xla program (torch-xla doesn't always close all processes, which prevents another program from starting):
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='pkill python; pkill pt_main'
```

## Customization

### Custom Training Logic

You can define custom training steps by subclassing [trainers/base_trainer.BaseTrainer](./src/trainers/base_trainer.py).

`train_step(batch)` takes in the current training batch, computes the loss, applies the gradient step, and returns the loss, a dictionary of auxiliary values, the gradient norm, and the learning rate.

`forward(**kwargs)` takes in batch arguments and defines the forward pass to compute the loss and aux values.

See [trainers/lm_trainer.py](./src/trainers/lm_trainer.py) and [trainers/seq_to_seq_lm_trainer.py](./src/trainers/seq_to_seq_lm_trainer.py) for examples.


### Custom Models

You can define custom models in the form of nn.Modules, just make sure not to have any graph breaks (printing values, .item() calls, host transfers).

Config files define model sharding, scanning, and checkpointing. See [configs/model/remat/](./src/configs/model/remat/) and [configs/model/sharding/](./src/configs/model/sharding) for examples.

Model checkpoints saved during training can be loaded on a non-xla device using the [models.load_checkpoint](./src/models/__init__.py) function.


### Custom Optimizers

Custom optimizers can be created using the torch.optim.Optimizer class. Initialization arguments are passed from the trainer config.

See [optimizers/adamw](./src/optimizers/adamw.py) for an example.


### Custom Data Loading

Data is loaded using the Huggingface datasets library, and easy-torch-tpu handles sending data to each device.

Custom data loading logic can be implemented using custom collator classes that convert raw batches into torch tensors.

See [collators/](./src/collators/) for examples.


### Configuration

There are a large number of configuration options. More information can be found in the [configs/](./src/configs/) folder.


## Training

To start a training run, run [train.py](./src/train.py) on all devices in the background (see above).

[Hydra](https://hydra.cc/docs/intro/) is used to manage configs. See the documentation for passing arguments through the command line.


## Acknowledgements

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC).
