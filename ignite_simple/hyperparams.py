"""This module is used to describe the hyperparameter tuning settings and
presets used for training."""

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
        which are equally spaced. Must be either 0 or greater than 1. If 0, the
        batch size corresponding to the greatest increase in accuracy during
        the reasonableness sweep is used.

    :ivar int batch_pt_min_inits: the minimum number of model initializations
        that are averaged together then smoothed for each batch point
    """
    def __init__(self, lr_start: float, lr_end: float, lr_min_inits: int,
                 batch_start: int, batch_end: int, batch_rn_min_inits: int,
                 batch_pts: int, batch_pt_min_inits: int):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_min_inits = lr_min_inits
        self.batch_start = batch_start
        self.batch_end = batch_end
        self.batch_rn_min_inits = batch_rn_min_inits
        self.batch_pts = batch_pts
        self.batch_pt_min_inits = batch_pt_min_inits

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
    )

def fast() -> HyperparameterSettings:
    """Returns a reasonably fast (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-6,
        lr_end=0.5,
        lr_min_inits=1,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=1,
        batch_pts=3,
        batch_pt_min_inits=1,
    )

def slow() -> HyperparameterSettings:
    """Returns a somewhat slow (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=0.5,
        lr_min_inits=3,
        batch_start=16,
        batch_end=128,
        batch_rn_min_inits=3,
        batch_pts=12,
        batch_pt_min_inits=3
    )

def slowest() -> HyperparameterSettings:
    """Returns the slowest (in time spent tuning parameters) preset"""
    return HyperparameterSettings(
        lr_start=1e-8,
        lr_end=0.5,
        lr_min_inits=10,
        batch_start=1,
        batch_end=512,
        batch_rn_min_inits=10,
        batch_pts=24,
        batch_pt_min_inits=10,
    )
