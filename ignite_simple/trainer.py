"""This module manages preparing and running the training environment for given
settings."""

import torch
import typing
import ignite.engine
import ignite.contrib.handlers.param_scheduler

class TrainSettings:
    """Describes the settings which ultimately go into a training session. This
    is intended to be trivially serializable, in that all attributes are built-
    ins that can be json serialized.

    :ivar str accuracy_style: one of the following constants:

    * 'classification': labels are integers which correspond to the class,
    outputs are one-hot encoded classes

    * 'multiclass': labels are one-hot encoded multi-class labels, outputs are
    the same

    * 'inv-loss': accuracy is not measured and inverse loss is used as the
    performance metric instead. for stability, instead of exactly inverse loss,
    :math:`\frac{1}{\text{loss} + 1e-6}` is used instead.

    :ivar tuple[str, str] model_loader: the tuple contains the module and
    corresponding attribute name for a function which returns the
    nn.Module to train. The module must have the calling convention
    `model(inp) -> out`

    :ivar tuple[str, str] task_loader: the tuple contains the module and
    corresponding attribute name for a function which returns
    :code:`(train_set, val_set, train_loader)`, each as described in
    TrainState.

    :ivar float lr_start`: the learning rate at the start of each cycle

    :ivar float lr_end: the learning rate at the end of each cycle

    :ivar int cycle_time_epochs: the number of epochs for the learning rate
    scheduler

    :ivar int epochs: the number of epochs to train for
    """
    def __init__(self,
                 accuracy_style: str,
                 model_loader: typing.Tuple[str, str],
                 task_loader: typing.Tuple[str, str],
                 lr_start: float,
                 lr_end: float,
                 cycle_time_epochs: int,
                 epochs: int):
        self.accuracy_style = accuracy_style
        self.model_loader = model_loader
        self.task_loader = task_loader
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.cycle_time_epochs = cycle_time_epochs
        self.epochs = epochs

class TrainState:
    """Describes the state which is passed as the second positional argument to
    each event handler, which contains generic information about the training
    session that may be useful.

    :ivar torch.nn.Module model: the model which is being trained

    :ivar torch.utils.data.Dataset train_set: the dataset which is used to
    train the model

    :ivar torch.utils.data.Dataset val_set: the dataset which is used to
    validate the models performance on unseen / held out data.

    :ivar torch.utils.data.DataLoader train_loader: the dataloader which is
    being used to generate batches from the train set to be passed into the
    model. This incorporates the batch size

    :ivar torch.optim.Optimizer optimizer: the optimizer which is used to
    update the parameters of the model

    :ivar int cycle_time_epochs: the number of epochs in a complete cycle
    of the learning rate, always even

    :ivar ignite.contrib.handlers.param_scheduler.CyclicalScheduler lr_scheduler:
    the parameter scheduler for the learning rate. Its instance values can be
    used to get the learning rate range and length in batches

    :ivar torch.nn.Module loss: the loss function, which accepts
    :code:`(input, target)` and returns a scalar which is to be minimized

    :ivar ignite.engine.Engine evaluator: the engine which can be used to
    gather metrics. Always has a :code:`'loss'` and :code:`'perf'` metric, but
    may or may not have an :code:`'accuracy'` metric.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_set: torch.utils.data.Dataset,
                 val_set: torch.utils.data.Dataset,
                 train_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 cycle_time_epochs: int,
                 lr_scheduler: ignite.contrib.handlers.param_scheduler.CyclicalScheduler,
                 loss: torch.nn.Module,
                 evaluator: ignite.engine.Engine):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.cycle_time_epochs = cycle_time_epochs
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.evaluator = evaluator
