# Customization

This framework allows you to easily implement custom logic.

## Trainers

To define custom training logic, add a new file to the [src/trainers/](../src/trainers) folder. Within that file, define a new subclass of [BaseTrainer](../src/trainers/base_trainer.py).

Your class must contain a `forward` function that takes key-word arguments representing the current training batch and returns the loss (a single-element tensor) and the auxiliary metrics (a dictionary containing single-element tensors and/or other objects). This function has access to the global configuration through `self.config`, the index of the current training step through `self.step`, and the model through `self.model`.

You can also override the `train_step(batch)` function which takes in the current training batch (as given by your collator), computes the loss using `forward`, and applies the gradient step. This function should return the loss, the aux dictionary, the gradient norm, and the learning rate.

Make sure that neither `forward` nor `train_step` contain operations that will break the XLA compilation graph, such as `print(Tensor)` and `Tensor.item()`.

See [src/trainers/lm_trainer.py](../src/trainers/lm_trainer.py) for an example.


## Models

You can define custom models in the [src/models/](../src/models/) folder using nn.Modules. 

Again, make sure that when your model is called during training it does not contain operations that will break the XLA compilation graph, such as `print(Tensor)` and `Tensor.item()`.

If you want to train your model, you will need to define sharding, scanning, and remat configs. See [configs/model/remat/](./src/configs/model/remat/) and [configs/model/sharding/](./src/configs/model/sharding/) for examples and more information.

Model checkpoints that are saved during training can be loaded on a non-xla device using the [models.load_checkpoint](./src/models/__init__.py) function.


## Optimizers

Custom optimizers can be created in the [src/optimizers/](../src/optimizers/) using the torch.optim.Optimizer class. Initialization arguments are passed from the trainer config.

Your optimizer must implement a `step` function that returns a dictionary of single-element tensors and/or other objects to be logged.

See [optimizers/adamw.py](./src/optimizers/adamw.py) for an example.


### Collators

Custom data loading is implemented using [collator](https://docs.pytorch.org/docs/stable/data.html#working-with-collate-fn) classes in the [src/collators/](../src/collators/) folder.

Your collator class should implement an `__init__` function, to which arguments defined in the config will be passed as key-word arguments. Your class must also implement a `__call__(batch)` function, which takes in a list of dictionaries (with each dictionary representing a single data point) and return a dictionary of tensors. The tensors that your collator returns should not be moved to the TPU device.

Data is otherwise loaded using the [Hugging Face datasets library](https://huggingface.co/docs/datasets), and easy-torch-tpu handles sending data to each device.

See [src/collators/](../src/collators/) for examples.
