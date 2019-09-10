"""Module initialization. See readme for this modules details. The general
purpose of the ignite_simple.gen_sweep module is to support sweeping over
arbitrary variables, i.e., number of hidden nodes in a recurrent network,
where at each step you may resweep the learning rate and batch size (using
any hyperparameter preset desired)."""

import ignite_simple
from ignite_simple.gen_sweep.sweeper import sweep

__all__ = ['sweep']
