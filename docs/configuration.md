# Configuration

This repo uses [Hydra](https://hydra.cc/docs/intro/) to manage configuration files and options.

Examples and exhaustive configuration options can be found in the [src/configs/](../src/configs/) folder. In this file we review the main structure.


# default

The default file at [src/configs/default.yaml](../src/configs/default.yaml) contains the highest level configuration options, such as the random seed, dtype settings, distributed mesh configuration, and wandb logging information.


## trainer

These settings relate to the training setup, such as the global batch size, the checkpoint interval, and whether to use mixed precision.

This is also where custom trainer configurations should be set.

### optimizer

Within the trainer config is the optimizer config, which denotes the optimizer type and kwargs.

### lr_scheduler

The lr_scheduler config, with type and kwarg options, also resides within the trainer config.


## data

These settings relate to the dataset and loading.

### dataset

Information about the HuggingFace dataset that will be used for training.

### collator

Settings, including type and kwargs, for the collator that will be used to load the data.


## model

These settings relate to the model.

At the base level are the model type, pretrained checkpoint loading options, and architecture configuration (`hidden_size`, `num_hidden_layers`, etc).

### remat

The remat section within the model config relates to things like activation checkpointing, layer scanning, and activation offloading.

See [src/configs/model/remat/torchprime.yaml](../src/configs/model/remat/torchprime.yaml) for detailed information.

### sharding

This config sets the parameter and activation sharding specs for the model.

See [docs/sharding.md](./sharding.md) for more information and [src/configs/model/sharding/llama-fsdp.yaml](../src/configs/model/sharding/llama-fsdp.yaml) for an example.