"""This acts as a potential runner for files, or as an import to reduce the
amount of boilerplate in a runner. Given a module which has a model() function,
dataset() function, and loss() function this uses argparse to fill in the rest
of the parameters to train() or reanalyze() as requested.

:Example:

.. code-block:: python

    # in file mymod.py
    import ignite_simple.helper
    import torch

    def model():
        pass # omitted, should return torch.nn.Module

    def dataset():
        pass # omitted, return train_set, val_set

    accuracy_style = 'multiclass'
    loss = torch.nn.MSELoss # any callable that returns a loss works

    if __name__ == '__main__':
        ignite_simple.helper(__name__)

.. code-block:: none

    > python3 -m mymod --help

"""
import ignite_simple
import os
import argparse
import importlib
import logging.config

def train(module, args):
    """Uses the given arguments from argparse to determine the arguments to
    ignite_simple.train

    :param module: module containing the model(), dataset(), loss(), and
        accuracy_style
    :param args: argparse result
    """
    mod = importlib.import_module(module)
    ignite_simple.train(
        (module, 'model', tuple(), dict()),
        (module, 'dataset', tuple(), dict()),
        (module, 'loss', tuple(), dict()),
        folder=os.path.join(args.folder, 'current'),
        hyperparameters=args.hparams,
        analysis=args.analysis,
        allow_later_analysis_up_to=args.analysis_up_to,
        accuracy_style=getattr(mod, 'accuracy_style'),
        trials=args.trials,
        is_continuation=args.is_continuation,
        history_folder=os.path.join(args.folder, 'history'),
        cores=args.cores,
        trials_strict=args.strict_trials
    )

def reanalyze(module, args):
    """Uses the given arguments from argparse to determine the arguments to
    ignite_simple.reanalyze

    :param module: module containing model(), dataset(), loss(), accuracy_style
    :param args: argparse result
    """
    mod = importlib.import_module(module)
    ignite_simple.analyze(
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder=os.path.join(args.folder, 'current'),
        settings=args.analysis,
        accuracy_style=getattr(mod, 'accuracy_style'),
        cores=args.cores)

def handle(module=None):
    """Uses the given module containing model(), dataset(), loss() and
    accuracy_style as the module for train() or analyze() with everything
    else determined by the command line arguments.

    :param module: module containing model(), dataset(), loss(), accuracy_style
    """
    parser = argparse.ArgumentParser(description='Simple model/dataset helper')
    parser.add_argument(
        '--folder', type=str, default=None,
        help='Where to store the output')
    parser.add_argument(
        '--hparams', type=str, default='fast',
        help='Level of hyperparameter tuning, one of \'fastest\', \'fast\', '
        + '\'slow\', and \'slowest\'')
    parser.add_argument(
        '--analysis', type=str, default='images-min',
        help='Level of analysis to perform, typically images-min or videos')
    parser.add_argument(
        '--analysis_up_to', type=str, default='videos',
        help='Level of analysis that will be possible without repeating '
        + 'trials')
    parser.add_argument(
        '--trials', type=int, default=1,
        help='Minimum number of trials to perform')
    parser.add_argument(
        '--not_continuation', action='store_true',
        help='If specified, trials will be archived if they exist first')
    parser.add_argument(
        '--cores', type=int, default=-1,
        help='Number of cores to use, default is all physical cores available')
    parser.add_argument(
        '--reanalyze', action='store_true',
        help='Instead of training, just perform analysis on existing trials')
    parser.add_argument(
        '--module', type=str,
        help='Which module to load the model, dataset, loss, and '
        + ' accuracy style from')
    parser.add_argument(
        '--strict_trials', action='store_true',
        help='Instead of using all available resources, just perform the '
             + 'specified number of trials'
    )
    parser.add_argument(
        '--loggercfg', type=str, default='logging.conf',
        help='The logging configuration file to use should it exist'
    )
    args = parser.parse_args()
    args.is_continuation = not args.not_continuation
    args.cores = args.cores if args.cores != -1 else 'all'
    args.module = args.module if args.module is not None else module

    if args.folder is None:
        args.folder = os.path.join('out', *args.module.split('.'))

    if args.module is None:
        raise ValueError('Module must be set from command line or '
                         + 'delegating file')

    if os.path.exists(args.loggercfg):
        logging.config.fileConfig(args.loggercfg)
    else:
        print('No logging configuration found - continuing without logging')

    if args.reanalyze:
        reanalyze(args.module, args)
    else:
        train(args.module, args)

if __name__ == '__main__':
    handle()
