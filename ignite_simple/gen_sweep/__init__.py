"""Module initialization. See readme for this modules details. The general
purpose of the ignite_simple.gen_sweep module is to support sweeping over
arbitrary variables, i.e., number of hidden nodes in a recurrent network,
where at each step you may resweep the learning rate and batch size (using
any hyperparameter preset desired)."""

import ignite_simple

import ignite_simple.gen_sweep.param_selectors as param_selectors
import ignite_simple.gen_sweep.sweeper as sweeper

sweep = sweeper.sweep

