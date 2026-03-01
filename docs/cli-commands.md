# CLI Commands

This section explains how to run commands on cloud TPU virtual machines using the gcloud cli.

## Basic Commands

Run a command on all devices within a node:

```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='<COMMAND>'
```

## Background Commands

Run a command (such as a training run) in the background so that it continues running after the terminal disconnects:
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='nohup <COMMAND> > command.log 2>&1 &'
```

Monitor the outputs of a background command:
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='tail -f command.log'
```

## Terminating torch-xla

torch-xla doesn't always close all processes after stopping, which prevents new programs from starting. Run this command to kill all torch-xla processes so that you can run a new program:
```
gcloud compute tpus tpu-vm ssh <NODE_ID> --worker=all --command='pkill python; pkill pt_main'
```

## Config Options

The [training script](../src/train.py) uses [Hydra](https://hydra.cc/docs/intro/) to manage configuration files and options. This allows you to overwrite training configuration from the command line when you launch a training run. Here is an example:
```
python ~/easy-torch-tpu/src/train.py trainer=lm-medium model=tinyllama-medium trainer.global_batch_size=128
```
