"""This module is used to describe the hyperparameter tuning settings and
presets used for training."""
import typing

class HyperparameterSettings:
    """Describes settings for tuning hyperparameters

    :ivar float lr_start: the smallest learning rate that is checked

    :ivar float lr_end: the largest learning rate that is checked

    :ivar int lr_min_inits: the minimum number of model initializations that
        are averaged together and then smoothed to get the lr-vs-accuracy plot.
        Note that when multiple physical cores are available they will be
        utilized since this process is well-suited to parallelization

    :ivar bool lr_width_only_gradients: if True, when deciding the best range
        for learning rate, we prefer the widest range over the greatest
        integral for the derivative. We still clip with the integral.

    :ivar int batch_start: the smallest batch size that is checked during the
        initial reasonableness sweep (a single pass)

    :ivar int batch_end: the largest batch size that is checked during the
        initial reasonableness sweep (a single pass)

    :ivar int batch_rn_min_inits: the minimum number of model initializations
        that are averaged together then smoothed to get the batch-vs-accuracy
        plot.

    :ivar int batch_pts: the number of different batch sizes which are checked,
        points sampled from a distribution weighted toward a higher first
        derivative of performance. Must be either 0 or greater than 1. If 0, the
        batch size corresponding to the greatest increase in accuracy during
        the reasonableness sweep is used.

    :ivar int batch_pt_min_inits: the minimum number of model initializations
        that are combined together via LogSumExp. We use LogSumExp instead of
        mean because we care more about the best performance than the most
        consistent performance when selecting batch size. If you want more
        motivation, we prefer a final accuracy of [0, 1, 1] over 3 trials to
        [2/3, 2/3, 2/3] even though the mean is the same

    :ivar bool rescan_lr_after_bs: if True, the learning rate is scanned once
        more after we tweak the batch size. otherwise, we use the same ratio
        of learning rate to batch size as we found in the first sweep.

    :ivar float warmup_lr: Decides the learning rate which is used to warmup
        models prior to a learning rate sweep. Should be fairly low. Has no
        effect if warmup_pts <= 0

    :ivar int warmup_batch: Decides the batch size which is used to warmup
        models prior to a learning rate sweep. Should be fairly high. Has no
        effect if warmup_pts <= 0

    :ivar union[int, float] warmup_pts: The number of points to warmup models prior to
        learning rate sweeps. Should be low for stable models and higher for
        unstable models. Use an int for points, float for epochs
    """
    def __init__(self, lr_start: float, lr_end: float, lr_min_inits: int,
                 batch_start: int, batch_end: int, batch_rn_min_inits: int,
                 batch_pts: int, batch_pt_min_inits: int,
                 rescan_lr_after_bs: bool, warmup_lr: float,
                 warmup_batch: int, warmup_pts: typing.Union[int, float],
                 lr_width_only_gradients: bool):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_min_inits = lr_min_inits
        self.lr_width_only_gradients = lr_width_only_gradients
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.batch_rn_min_inits = batch_rn_min_inits
        self.batch_pts = batch_pts
        self.batch_pt_min_inits = batch_pt_min_inits
        self.rescan_lr_after_bs = rescan_lr_after_bs
        self.warmup_lr = warmup_lr
        self.warmup_batch = warmup_batch
        self.warmup_pts = warmup_pts

    def __repr__(self):
        return f'HyperparameterSettings(**{self.__dict__})'

def fastest() -> HyperparameterSettings:
    """Returns the fastest (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-6,
        lr_end=1,
        lr_min_inits=1,
        lr_width_only_gradients=False,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=1,
        batch_pts=0,
        batch_pt_min_inits=0,
        rescan_lr_after_bs=False,
        warmup_lr=1e-6,
        warmup_batch=64,
        warmup_pts=0.1,
    )

def fast() -> HyperparameterSettings:
    """Returns a reasonably fast (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-6,
        lr_end=1,
        lr_min_inits=1,
        lr_width_only_gradients=False,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=1,
        batch_pts=3,
        batch_pt_min_inits=1,
        rescan_lr_after_bs=False,
        warmup_lr=1e-6,
        warmup_batch=64,
        warmup_pts=0.1,
    )

def slow() -> HyperparameterSettings:
    """Returns a somewhat slow (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=1,
        lr_min_inits=3,
        lr_width_only_gradients=False,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=3,
        batch_pts=12,
        batch_pt_min_inits=3,
        rescan_lr_after_bs=True,
        warmup_lr=1e-6,
        warmup_batch=64,
        warmup_pts=0.1,
    )

def slowest() -> HyperparameterSettings:
    """Returns the slowest (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=1,
        lr_min_inits=10,
        lr_width_only_gradients=False,
        batch_start=8,  # <8 might sometimes be better, but is painfully
        batch_end=512,  # slow computationally
        batch_rn_min_inits=10,
        batch_pts=24,
        batch_pt_min_inits=10,
        rescan_lr_after_bs=True,
        warmup_lr=1e-6,
        warmup_batch=64,
        warmup_pts=0.1,
    )

NAME_TO_PRESET = {
    'fastest': fastest,
    'fast': fast,
    'slow': slow,
    'slowest': slowest
}

def get_settings(preset: typing.Union[str, HyperparameterSettings]):
    """Gets the corresponding preset if the argument is a name of one,
    returns the argument directly if the argument is already a settings
    object, and raises an exception in all other circumstances.

    :param preset: the name for a preset or the complete settings object

    :returns: the corresponding preset or the settings object passed in
    """
    if isinstance(preset, HyperparameterSettings):
        return preset
    return NAME_TO_PRESET[preset]()
