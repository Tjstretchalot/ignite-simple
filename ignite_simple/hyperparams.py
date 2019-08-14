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
    """
    def __init__(self, lr_start: float, lr_end: float, lr_min_inits: int,
                 batch_start: int, batch_end: int, batch_rn_min_inits: int,
                 batch_pts: int, batch_pt_min_inits: int,
                 rescan_lr_after_bs: bool):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_min_inits = lr_min_inits
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.batch_rn_min_inits = batch_rn_min_inits
        self.batch_pts = batch_pts
        self.batch_pt_min_inits = batch_pt_min_inits
        self.rescan_lr_after_bs = rescan_lr_after_bs

def fastest() -> HyperparameterSettings:
    """Returns the fastest (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-4,
        lr_end=0.3,
        lr_min_inits=1,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=1,
        batch_pts=0,
        batch_pt_min_inits=0,
        rescan_lr_after_bs=False,
    )

def fast() -> HyperparameterSettings:
    """Returns a reasonably fast (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-6,
        lr_end=1,
        lr_min_inits=1,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=1,
        batch_pts=3,
        batch_pt_min_inits=1,
        rescan_lr_after_bs=False,
    )

def slow() -> HyperparameterSettings:
    """Returns a somewhat slow (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=1,
        lr_min_inits=3,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=3,
        batch_pts=12,
        batch_pt_min_inits=3,
        rescan_lr_after_bs=True,
    )

def slowest() -> HyperparameterSettings:
    """Returns the slowest (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=1,
        lr_min_inits=10,
        batch_start=8,  # <8 might sometimes be better, but is painfully
        batch_end=512,  # slow computationally
        batch_rn_min_inits=10,
        batch_pts=24,
        batch_pt_min_inits=10,
        rescan_lr_after_bs=True,
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
