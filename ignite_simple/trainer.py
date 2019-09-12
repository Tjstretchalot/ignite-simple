"""This module manages preparing and running the training environment for given
settings."""

import torch
import torch.utils.data
import typing
from ignite_simple.utils import noop
import ignite.engine
import ignite.contrib.handlers.param_scheduler
import importlib
import functools

class TrainSettings:
    r"""Describes the settings which ultimately go into a training session.
    This is intended to be trivially serializable, in that all attributes are
    built-ins that can be json serialized. The train function here runs in the
    same process, but this strategy allows us to use the same interface design
    throughout and allows repeating / printing training sessions trivially.

    :ivar str accuracy_style: one of the following constants:

        * classification
            labels are one-hot encoded classes,
            outputs are one-hot encoded classes.
        * multiclass
            labels are one-hot encoded multi-class labels, outputs are the same
        * inv-loss
            accuracy is not measured and inverse loss is used as the
            performance metric instead. For stability, and legibility of plots,

            .. math::
                \frac{1}{\text{loss} + 1}

            is used.

    :ivar tuple[str, str, tuple, dict] model_loader: the tuple contains the
        module and corresponding attribute name for a function which returns
        the nn.Module to train. The module must have the calling convention
        `model(inp) -> out`. The next two arguments are the args and keyword
        args to the callable respectively.

    :ivar tuple[str, str, tuple, dict] loss_loader: the tuple contains the
        model and corresponding attribute name for a function which returns
        the loss function to minimize.

    :ivar tuple[str, str, tuple, dict] task_loader: the tuple contains the
        module and corresponding attribute name for a function which returns
        :code:`(train_set, val_set, train_loader)`, each as described in
        TrainState. The next two arguments are the args and keyword args
        to the callable respectively.

    :ivar tuple[tuple[str, tuple[str, str, tuple, dict]]] handlers: the event
        handlers for the engine which will perform training. After the
        specified positional arguments, the handlers will be passed the Engine
        that is training the model and the TrainState that is in use. The str
        associated with each callable is the event that each callable listens
        to.

        .. code::python

            import ignite.engine as engine

            def log_epoch(format, tnr, state):
                print(format.format(tnr.state.epoch))

            handlers = (
                (engine.Event.EPOCH_COMPLETED,
                 (__name__, 'log_epoch', ('Completed Epoch {}',), dict())),
            )

            # handlers is suitable for this variable now, so long as the
            # __name__ was not __main__

    :ivar tuple[str, str, tuple, dict] initializer: this is called with trainer
        as the next positional argument. May be used to attach additional
        events to the trainer.

    :ivar float lr_start: the learning rate at the start of each cycle

    :ivar float lr_end: the learning rate at the end of each cycle

    :ivar int cycle_time_epochs: the number of epochs for the learning rate
        scheduler

    :ivar int epochs: the number of epochs to train for
    """
    def __init__(
            self,
            accuracy_style: str,
            model_loader: typing.Tuple[str, str, tuple, dict],
            loss_loader: typing.Tuple[str, str, tuple, dict],
            task_loader: typing.Tuple[str, str, tuple, dict],
            handlers: typing.Tuple[
                typing.Tuple[str, typing.Tuple[str, str, tuple, dict]]],
            initializer: typing.Optional[typing.Tuple[str, str, tuple, dict]],
            lr_start: float,
            lr_end: float,
            cycle_time_epochs: int,
            epochs: int):
        self.accuracy_style = accuracy_style
        self.model_loader = model_loader
        self.loss_loader = loss_loader
        self.task_loader = task_loader
        self.handlers = handlers
        self.initializer = initializer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.cycle_time_epochs = cycle_time_epochs
        self.epochs = epochs

    def get_model_loader(self) -> typing.Callable:
        """Gets the actual model loader callable, which is defined through
        model_loader. This is a callable which returns the `torch.nn.Module`
        to train. The resulting callable already has the required arguments
        and keyword arguments bound.
        """
        module = importlib.import_module(self.model_loader[0])
        func = getattr(module, self.model_loader[1])
        return functools.partial(
            func, *self.model_loader[2], **self.model_loader[3])

    def get_loss_loader(self) -> typing.Callable:
        """Gets the actual loss loader callable, which is defined through
        loss_loader. This is a callable which returns the `torch.nn.Module`
        that goes from the output of the model to a scalar which should be
        minimized."""
        module = importlib.import_module(self.loss_loader[0])
        func = getattr(module, self.loss_loader[1])
        return functools.partial(
            func, *self.loss_loader[2], **self.loss_loader[3])

    def get_task_loader(self) -> typing.Callable:
        """Gets the actual task loader callable, which is defined through
        task_loader. This is a callable which returns
        `(train_set, val_set, train_loader)`, each as defined in TrainState.
        The resulting callable already has the required arguments
        and keyword arguments bound.
        """
        module = importlib.import_module(self.task_loader[0])
        func = getattr(module, self.task_loader[1])
        return functools.partial(
            func, *self.task_loader[2], **self.task_loader[3])

    def get_handlers(self) -> typing.Tuple[typing.Tuple[str, typing.Callable]]:
        """This returns handlers except instead of the function descriptions
        (module, attribute, args, kwargs), actual callables are provided with
        the necessary arguments and keyword arguments already bound.
        """
        res = []
        for evt, (modnm, attrnm, args, kwargs) in self.handlers:
            module = importlib.import_module(modnm)
            func = getattr(module, attrnm)
            func = functools.partial(func, *args, **kwargs)
            res.append((evt, func))
        return tuple(res)

    def get_initializer(self) -> typing.Callable:
        """This returns the initializer; if it is not specified this is a
        no-op. Otherwise, this is the callable which accepts the trainer
        and initializes it, with the other arguments and keyword arguments
        already bound."""
        if not self.initializer:
            return noop

        module = importlib.import_module(self.initializer[0])
        func = getattr(module, self.initializer[1])
        return functools.partial(
            func, *self.initializer[2], **self.initializer[3])

class TrainState:
    """Describes the state which is passed as the second positional argument to
    each event handler, which contains generic information about the training
    session that may be useful.

    :ivar torch.nn.Module model: the model which is being trained

    :ivar optional[torch.nn.Module] unstripped_model: the unstripped model, if
        there is one, otherwise just the same reference as model

    :ivar torch.utils.data.Dataset train_set: the dataset which is used to
        train the model

    :ivar torch.utils.data.Dataset val_set: the dataset which is used to
        validate the models performance on unseen / held out data.

    :ivar torch.utils.data.DataLoader train_loader: the dataloader which is
        being used to generate batches from the train set to be passed into the
        model. This incorporates the batch size.

    :ivar torch.optim.Optimizer optimizer: the optimizer which is used to
        update the parameters of the model.

    :ivar int cycle_time_epochs: the number of epochs in a complete cycle
        of the learning rate, always even.

    :ivar ignite.contrib.handlers.param_scheduler.CyclicalScheduler lr_scheduler:
        the parameter scheduler for the learning rate. Its instance values can
        be used to get the learning rate range and length in batches.

    :ivar torch.nn.Module loss: the loss function, which accepts
        :code:`(input, target)` and returns a scalar which is to be minimized.

    :ivar ignite.engine.Engine evaluator: the engine which can be used to
        gather metrics. Always has a :code:`'loss'` and :code:`'perf'` metric, but
        may or may not have an :code:`'accuracy'` metric.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 unstripped_model: typing.Optional[torch.nn.Module],
                 train_set: torch.utils.data.Dataset,
                 val_set: torch.utils.data.Dataset,
                 train_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 cycle_time_epochs: int,
                 lr_scheduler: ignite.contrib.handlers.param_scheduler.CyclicalScheduler,
                 loss: torch.nn.Module,
                 evaluator: ignite.engine.Engine):
        self.model = model
        self.unstripped_model = unstripped_model
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.cycle_time_epochs = cycle_time_epochs
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.evaluator = evaluator

def _multilabel_threshold(output):
    y_pred, y = output
    y_pred = y_pred.clone()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    return y_pred, y

def _singlelabel_threshold(output):
    y_pred, y = output[0].detach(), output[1].detach()
    ny_pred = torch.zeros_like(y_pred)
    ny_pred[torch.arange(y_pred.shape[0]), y_pred.argmax(1)] = 1
    ny = y if len(y.shape) == 1 else y.argmax(1)
    return ny_pred, ny

def _inv_loss(loss):
    return 1 / (loss + 1)

def _iden(x):
    return x

def train(settings: TrainSettings) -> None:
    """Trains a model with the given settings.

    .. note::

        In order to store anything you will need to use a handler. For example,
        a handler for `ignite.engine.Event.COMPLETED` and stores the
        model somewhere.

    :param TrainSettings settings: The settings to use for training
    """
    model = settings.get_model_loader()()
    loss = settings.get_loss_loader()()
    train_set, val_set, train_loader = settings.get_task_loader()()

    if isinstance(model, tuple):
        unstripped_model = model[0]
        model = model[1]
    else:
        unstripped_model = model

    handlers = settings.get_handlers()

    metrics = {'loss': ignite.metrics.Loss(loss)}
    if settings.accuracy_style == 'classification':
        metrics['accuracy'] = ignite.metrics.Accuracy(_singlelabel_threshold)
        metrics['perf'] = ignite.metrics.MetricsLambda(
            _iden, metrics['accuracy'])
    elif settings.accuracy_style == 'multiclass':
        metrics['accuracy'] = ignite.metrics.Accuracy(_multilabel_threshold,
                                                      is_multilabel=True)
        metrics['perf'] = ignite.metrics.MetricsLambda(
            _iden, metrics['accuracy'])
    else:
        metrics['perf'] = ignite.metrics.MetricsLambda(
            _inv_loss, metrics['loss'])

    optimizer = torch.optim.SGD(model.parameters(), lr=1)  # lr irrelevant here
    scheduler = (
        ignite.contrib.handlers.param_scheduler.LinearCyclicalScheduler(
            optimizer, 'lr', settings.lr_start, settings.lr_end,
            len(train_loader) * settings.cycle_time_epochs
        )
    )

    trainer = ignite.engine.create_supervised_trainer(model, optimizer, loss)
    evaluator = ignite.engine.create_supervised_evaluator(
        model, metrics=metrics)

    settings.get_initializer()(trainer)

    state = TrainState(
        model, unstripped_model, train_set, val_set, train_loader, optimizer,
        settings.cycle_time_epochs, scheduler, loss, evaluator
    )

    trainer.add_event_handler(
        ignite.engine.Events.ITERATION_STARTED,
        scheduler
    )

    for evt, hndlr in handlers:
        trainer.add_event_handler(evt, hndlr, state)

    trainer.run(train_loader, max_epochs=settings.epochs)
