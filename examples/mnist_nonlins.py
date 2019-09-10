
"""Investigate the effect of various nonlinearities on a 2-layer network
in MNIST
"""
import ignite_simple.gen_sweep as gen_sweep
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

    res = gen_sweep.sweep(
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
