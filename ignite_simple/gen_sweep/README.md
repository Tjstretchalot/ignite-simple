# ignite_simple.gen_sweep

This module handles sweeping over generic user-defined parameters. Typically
this is for architecture variables, such as the number of hidden nodes in a
neural network or the depth of a convolution network, etc. This process is
painfully slow in general, but this is meant to allow throwing as much (CPU)
power as is available at the problem.

This type of parallelization is perhaps not a beneficial on a GPU where
operations are already parallelized, but most individuals do not have access
to a GPU powerful / recent enough for machine learning.

## Example

The following can be used to understand the effect of a nonlinearity on
training a small 2-layer network on MNIST

```py
"""Investigate the effect of various nonlinearities on a 2-layer network
in MNIST
"""
import ignite_simple.gen_sweep.sweeper as sweeper
from ignite_simple.gen_sweep.param_selectors import FixedSweep
import torch
from torchluent import FluentModule
import torchvision

NONLINEARITIES = ['Tanh', 'ReLU', 'PReLU']

def model(nonlin: str):
    return (
        FluentModule((1, 28, 28))
        .flatten()
        .dense(128)
        .operator(nonlin)
        .dense(10)
        .build()
    )

def dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss
accuracy_style = 'classification'

def main():
    import os
    import json
    import logging
    import logging.config
    import matplotlib.pyplot as plt
    import numpy as np

    folder = os.path.join('out', 'mnist-nonlins')
    os.makedirs(folder) # fail if already exists

    if os.path.exists('logging-gen.conf'):
        logging.config.fileConfig('logging-gen.conf')
    elif os.path.exists('logging.conf'):
        logging.config.fileConfig('logging.conf')
    else:
        print('No logging file found! Either cancel with Ctrl+C or press')
        print('enter to continue with no output.')
        input()

    res = sweeper.sweep(
        __name__,  # the module which contains the model/dataset/loss/accuracy
        FixedSweep.with_fixed_trials(  # Trial selector
            list((s,) for s in NONLINEARITIES),  # one variable
            3,  # trials per nonlinearity
        ),
        3,  # cores to use on hparam sweep. if exceeds number of physical cores,
            # the same amount of work is done as if you had this many
            # cores, but parallelized appropriately for your computer.
            # Determines minimum number of model initializations for LR
            # and BS sweeps. If there are 8 physical cores available and
            # this value is 3, then we can sweep hparams for 2 points
            # at the same time while also performing up to 2 trials
            # on other points
        'fast',  # hyperparameter selection preset
        folder
    )

    # res is a list which might look like:
    # [(('tanh',), 0.1, 1, 16, np.array((0.8, 0.85, 0.75)),
    #   np.array((0.1, 0.08, 0.12)), np.array((0.76, 0.82, 0.72)),
    #   np.array((0.12, 0.1, 0.14))), ...]
    # more information is stored in out/mnist-nonlins/points
    # res can be loaded later by unpickling out/mnist-nonlins/points/funcres.pt

    labels = [pt[0][0] for pt in res]
    index = np.arange(len(labels))
    avg_perfs = [pt[4].mean() for pt in res]  # use 6 for validation

    fig, ax = plt.subplots()
    ax.set_title('Performance by nonlinearity')
    ax.set_xlabel('Nonlinearity')
    ax.set_ylabel('Accuracy (%) (Train)')
    ax.bar(index, avg_perfs)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'result.pdf'))

    plt.close(fig)

if __name__ == '__main__':
    main()
```

## Requirements for parameters

Parameters must support equality and be hashable (i.e., implement
\_\_hash\_\_), strable (i.e., implement \_\_str\_\_) and picklable.

## Procedure

Given variables $x_1, ..., x_n$ and a sampling function `sampler` which is
given the points already selected and metrics for those points, the sampler
returns the values for the variables to test and the number of trials to
perform. The sampler must be able to return an arbitrary number of points
by request at once in order to make use of all available CPU cores.

Then we pass those variables to another given function `model`, which produces
a `torch.nn.Module` from those parameters. We are also given a `dataset`
function and `accuracy-style`, which allows the `ignite_simple.gen_sweep`
module to perform learning rate and batch-size tuning much like the
`ignite_simple` module does for that particular model.

We proceed with this until we have sampled the desired number of points.
This gives a list where each item is of the form

$(x_1, ..., x_n)$, final min learning rate, final max learning rate, final batch size,
    performance by trial, loss by trial

which is stored. In the special case where $n=1$, this can be plots where the
y-axis is performance and the x-axis is that variable are possible. Grid
searches also lend themselves to making multiple versions of these plots.

## Provided point selection methods

- `ignite_simple.param_selectors.FixedSweep` - given a set of points to
    check, check them each in an arbitrary order. This has class
    methods to produce the same number of trials over a set of points
    or a random selection of points from intervals or sets.
