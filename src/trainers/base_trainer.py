"""Base trainer module for TPU-based model training using PyTorch/XLA.

This script provides a `Trainer` class that sets up model sharding, activation checkpointing,
optimization, and the training loop with XLA-specific configurations. It is designed to work with
distributed TPU training and includes utilities for metrics logging and MFU computation.

Typical usage example:

    trainer = Trainer(model, config, train_dataset)
    trainer.train_loop(metrics_logger)
"""

import logging
import math
import os
from timeit import default_timer as timer
import shutil
import json
import numpy as np
import re
import time

import torch
import torch.nn.utils as nn_utils
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
from transformers import (
    get_scheduler,
)

from torchprime.torch_xla_models.model_rewriting.assume_pure import (
    mark_pure_modules,
)
from torchprime.torch_xla_models.model_rewriting.auto_trace import auto_trace
from torchprime.torch_xla_models.model_rewriting.sharding_initialization import (
    setup_sharding_and_mesh,
)
from torchprime.utils.parallelism_utils import lb_cp_enabled, reorder_sequence

import wandb
import huggingface_hub as hf

from models.xla import BaseXLAModel
from utils.import_utils import import_optimizer, import_collator
from utils import constants
from utils.remat_utils import advanced_remat
from utils.git_utils import get_current_commit_hash


logger = logging.getLogger(__name__)


class BaseTrainer:
    """Trainer class for TPU-accelerated model training using PyTorch/XLA.

    This class encapsulates model preparation, optimizer configuration, data loading,
    and the training loop. It is designed to handle distributed training across TPU cores,
    enabling features like SPMD sharding, activation checkpointing, and profiling.

    Args:
        model: The model to train.
        config: Configuration object containing training hyperparameters and setup.
        train_dataset: Dataset used for training.
    """

    minibatch: bool

    def __init__(
        self,
        model: BaseXLAModel,
        config: DictConfig,
        train_dataset: Dataset | IterableDataset,
    ):
        self.config = config
        self.device = xm.xla_device()
        
        self.global_batch_size = self.config.trainer.global_batch_size
        self.train_dataset = train_dataset

        self.model = self.prepare_model(model, config)

        self.optimizer, self.lr_scheduler = self.prepare_optimization(self.model, config)

        # set up saving
        if not self.config.debug and constants.PROCESS_IS_MAIN():
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            # create the huggingface save repo
            self.repo_name = f"{constants.HF_ID}/{self.config.project}_{self.config.name}"

            hf.create_repo(
                self.repo_name, private=False, exist_ok=True
            )

            # create the wandb project
            notes = f"GIT HASH: {get_current_commit_hash()}"
            if self.config.notes is not None:
                notes += f"\n\n{self.config.notes}"
            wandb.init(
                project=self.config.project,
                name=self.config.name,
                notes=notes,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        self.post_init()

        # Execute all initialization work queued so far before starting training.
        torch_xla.sync()


    def post_init(self):
        return


    def prepare_model(
        self, model, config: DictConfig
    ):
        """ Prepares the model for training by setting up sharding and rematerialization. """
        
        # Recursively replace `nn.Linear` layers with einsum operations in the model.
        # Without this patch, an `nn.Linear` module will flatten non-contracting dimensions
        # (e.g. batch and sequence), thus destroying the sharding constraints on those dimensions.
        model = apply_xla_patch_to_nn_linear(model)

        # Add `xp.Trace` to linear layers in the module tree (just for profiling?).
        model = auto_trace(model)

        # print model parameters that to not have sharding spec
        config_names = set(self.config.model.sharding.keys())
        param_names = set()
        for name, p in model.named_parameters():
            if p is not None:
                param_names.add(re.sub("\.\d+\.", ".*.", name))
        all_found = True
        for name in param_names:
            if name not in config_names:
                logger.warning(f"Parameter {name} does not have sharding spec!")
                all_found = False
        if all_found:
            logger.info("All model parameters have sharding spec.")

        # Setup SPMD mesh and shard the model.
        model, self.input_sharding_spec, self.minibatch = setup_sharding_and_mesh(
            model, config
        )
        model = mark_pure_modules(model, config)

        model = advanced_remat(model, config)

        return model


    def get_trainable_parameters(self, model: torch.nn.Module):
        return model.parameters()


    def prepare_optimization(
        self,
        model: torch.nn.Module,
        config: DictConfig
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """ Sets up the optimizer and learning rate scheduler. """

        params = self.get_trainable_parameters(model)

        optimizer = import_optimizer(config.trainer.optimizer.type)(
            params=params,
            **config.trainer.optimizer.kwargs,
        )

        lr_scheduler = get_scheduler(
            name=config.trainer.lr_scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=config.trainer.lr_scheduler.num_warmup_steps,
            scheduler_specific_kwargs=config.trainer.lr_scheduler.kwargs,
        )

        return optimizer, lr_scheduler
    

    def _get_train_dataloader(self) -> pl.MpDeviceLoader:

        num_replicas = xr.process_count()
        logger.info("Num replicas: %d", num_replicas)

        if self.minibatch:
            # Each process loads the per-host batch size.
            batch_size = self.global_batch_size // num_replicas
        else:
            # Each process will load the global batch, then discard the unneeded parts.
            batch_size = self.global_batch_size

        # handle the collator
        collator = import_collator(self.config.data.collator.type)(
            **self.config.data.collator.kwargs
        )
        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        loader = pl.MpDeviceLoader(
            dataloader, self.device, input_sharding=self.input_sharding_spec
        )
        
        return loader
    

    def save_checkpoint(
        self,
        step: int,
    ):
        logger.info("[SAVING] Starting distributed checkpoint...")

        # wait for existing operations
        xm.wait_device_ops()
        xm.rendezvous(f"checkpoint_start_{step}")

        # move the model to CPU for saving
        logger.info("Moving model to CPU for checkpoint saving...")
        state = {
            k: xs.clear_sharding(v.clone()).detach().cpu()
            for k, v in self.model.state_dict().items()
        }
        xm.mark_step()
        xm.wait_device_ops()
        xm.rendezvous(f"checkpoint_model_moved_{step}")
        logger.info("Moded model to CPU for checkpoint saving.")

        if constants.PROCESS_IS_MAIN(): 

            save_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                "tmp_checkpoint",
            )
            shutil.rmtree(save_path, ignore_errors=True)
            os.makedirs(save_path, exist_ok=True)

            logger.info(f"Saving config to {save_path}")
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump(OmegaConf.to_container(self.config.model, resolve=True), f, indent=4)
            logger.info(f"Saved config to {save_path}/config.json")

            logger.info(f"Saving model state to {save_path}")
            torch.save(state, os.path.join(save_path, "model.pt"))
            # np_state = {k: v.numpy() for k, v in state.items()}
            # print({k: np.reshape(v, (-1,))[:10] for k, v in np_state.items()})
            # np.save(os.path.join(save_path, "model.npy"), np_state)
            logger.info(f"Saved model state to {save_path}/model.pt")
            
            api = hf.HfApi()
            out_path = f"{step:012d}"

            logger.info(f"Uploading checkpoint to {self.repo_name}")
            api.upload_folder(
                repo_id=self.repo_name,
                folder_path=save_path,
                path_in_repo=out_path,
                repo_type="model",
            )
            logger.info(f"Uploaded checkpoint to {self.repo_name}/{out_path}")

            shutil.rmtree(save_path, ignore_errors=True)
        
        xm.rendezvous(f"checkpoint_saved_{step}")
        logger.info("[SAVING] Finished distributed checkpoint.")      
    

    def train_loop(self) -> None:

        # prepare model for training
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.get_trainable_parameters(self.model):
            p.requires_grad_(True)
        self.model.train()
        self.model.zero_grad()

        # prepare data loader
        max_step = self.config.trainer.max_steps
        train_loader = self._get_train_dataloader()
        train_iterator = iter(train_loader)

        # print training information
        logger.info("Starting training")
        logger.info("    Max step: %d", max_step)
        logger.info("    Global batch size: %d", self.global_batch_size)
        logger.info(f"    Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"    Model dtype: {list(self.model.parameters())[0].dtype}")

        # initialize counters
        epoch = 0 # TODO: currently hangs on new epoch
        step = 0
        error_count = 0
        self.atoms_seen = 0 # must be self. for step_closure
        training_start_time = timer()

        # run the training loop
        while step < max_step:
            try:
                batch = next(train_iterator)
            except:
                logger.warning("Unexpected error when fetching data at step %d, retrying", step)
                error_count += 1
                if error_count > 1000:
                    logger.error("Too many errors when fetching data, saving checkpoint and exiting")
                    self.save_checkpoint(step)
                    raise RuntimeError("Too many errors when fetching data")
                time.sleep(1)
                continue
            error_count = 0
            
            # skip steps for pretrained model
            if self.config.model.pretrained_step is not None and step < self.config.model.pretrained_step:
                if step % 10 == 0:
                    logger.info(f"Skipping step {step} as it is before the pretrained step {self.config.model.pretrained_step}")
                step += 1
                continue

            # can be reached by forward
            self.step = step

            # when context parallel and load balance context parallel is enabled,
            # we will reorder the sequence here for each batch
            if lb_cp_enabled(self.config):
                return {
                    key: reorder_sequence(
                        tensor=value,
                        cp_size=self.config.ici_mesh.context,
                        seq_dim=1,
                        to_contiguous=False,
                    )
                    for key, value in batch.items()
                }

            # perform the training step
            trace_start_time = timer()
            loss, aux, grad_norm, lr = self.train_step(batch)
            trace_end_time = timer()

            # post-step closure for logging
            def step_closure(
                start_time, epoch, step, loss, grad_norm, aux, trace_start_time, trace_end_time, lr
            ):
                training_time_elapsed = (
                    timer() - start_time  
                ) / 3600 # in hours

                if "atom_count" in aux.keys():
                    self.atoms_seen += aux["atom_count"].detach().item()

                loss = loss.detach().item()
                grad_norm = grad_norm.detach().item()

                logger.info(
                    "Hours elapsed: %.3f, epoch: %d, step: %d, loss: %.3f, grad_norm: %.3f, trace time: %.0f ms",
                    training_time_elapsed,
                    epoch,
                    step,
                    loss,
                    grad_norm,
                    (trace_end_time - trace_start_time) * 1000,
                )

                to_wandb = {}
                for k, v in aux.items():
                    if isinstance(v, torch.Tensor):
                        to_wandb[k] = v.detach().item()
                    else:
                        to_wandb[k] = v

                to_wandb["loss"] = loss
                to_wandb["grad_norm"] = grad_norm
                to_wandb["trace_time_ms"] = (trace_end_time - trace_start_time) * 1000
                to_wandb["lr"] = lr
                to_wandb["epoch"] = epoch

                to_wandb["examples_seen"] = (step + 1) * self.global_batch_size
                if "atom_count" in aux.keys():
                    to_wandb["atoms_seen"] = self.atoms_seen

                to_wandb["nan"] = 1 - int(math.isfinite(loss))

                to_wandb["training_time_elapsed_hr"] = training_time_elapsed
                to_wandb["avg_time_per_step_s"] = training_time_elapsed * 3600 / (step + 1)
                to_wandb["avg_steps_per_hr"] = (step + 1) / training_time_elapsed

                if not self.config.debug and constants.PROCESS_IS_MAIN():
                    wandb.log(to_wandb, step=step)
            
            # execute
            xm.add_step_closure(
                step_closure,
                args=(
                    training_start_time,
                    epoch,
                    step,
                    loss.detach().clone(),
                    grad_norm.detach().clone(),
                    {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in aux.items()},
                    trace_start_time,
                    trace_end_time,
                    lr,
                ),
                run_async=True,
            )
            xm.mark_step()

            # save checkpoint
            if (step+1) % self.config.trainer.checkpoint_interval == 0:    
                self.save_checkpoint(step+1)

            step += 1

        xm.wait_device_ops()
        logger.info("Finished training run")


    @torch_xla.compile(full_graph=True)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        
        loss, aux = self.forward(**batch)

        loss.backward()
        
        grad_norm = self.clip_gradients()

        opt_aux = self.optimizer.step()
        aux.update(opt_aux)
        self.model.zero_grad(set_to_none=False)

        lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()

        return loss, aux, grad_norm, lr


    def forward(self, **batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError(
            "The forward method should be implemented in the derived class."
        )


    def clip_gradients(self):
        """Clip gradients by the specified max norm and/or max absolute value."""
        max_grad_norm = self.config.trainer.max_grad_norm
        if max_grad_norm is None or max_grad_norm <= 0:
            grad_norm = nn_utils.get_total_norm(self.model.parameters(), norm_type=2)
        else:
            grad_norm = nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_grad_norm, norm_type=2
            )
        max_grad_value = self.config.trainer.max_grad_value
        if max_grad_value is not None and max_grad_value > 0:
            nn_utils.clip_grad_value_(self.model.parameters(), clip_value=max_grad_value)
        return grad_norm
