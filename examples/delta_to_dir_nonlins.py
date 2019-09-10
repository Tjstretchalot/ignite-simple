"""Investigates the effect of a nonlinearity on the toy delta to dir example.
This is meant to be analagous to mnist_nonlins except actually completes
in a reasonable amount of time on a consumer laptop
"""
import ignite_simple
import ignite_simple.gen_sweep.sweeper as sweeper
from ignite_simple.gen_sweep.param_selectors import FixedSweep
import torch
from torchluent import FluentModule
import os

NONLINEARITIES = ['Tanh', 'ReLU', 'PReLU']

def model(nonlin: str):
    """Creates the model that should be trained"""
    return (
        FluentModule((2,))
        .flatten()
        .dense(128)
        .operator(nonlin)
        .dense(4)
        .build()
    )

def dataset(max_abs_val: int = 30, build=False):
    """Creates a dataset that has the given maximum absolute value
    for the deltas

    Args:
        max_abs_val (int, optional): maximum values for each component. Defaults to 30.
    """
    if not build and os.path.exists('datasets/delta_to_dir.pt'):
        return torch.load('datasets/delta_to_dir.pt')

    side_len = max_abs_val * 2 + 1
    num_pts = side_len * side_len

    inps = torch.zeros((num_pts, 2), dtype=torch.float)
    outs = torch.zeros((num_pts, 4), dtype=torch.float)

    ind = 0
    for y in range(-max_abs_val, max_abs_val + 1):
        inps[ind:ind + side_len, 0] = torch.arange(
            -max_abs_val, max_abs_val + 1)
        inps[ind:ind + side_len, 1] = y

        if y < 0:
            outs[ind:ind + side_len, 0] = 1
        elif y > 0:
            outs[ind:ind + side_len, 2] = 1

        outs[ind:ind + max_abs_val, 1] = 1
        outs[ind + max_abs_val + 1:ind + side_len, 3] = 1
        ind += side_len

    prm = torch.randperm(num_pts)
    inps = inps[prm].contiguous()
    outs = outs[prm].contiguous()

    dset = torch.utils.data.TensorDataset(inps, outs)

    result = ignite_simple.utils.split(dset, 0.1)
    os.makedirs('datasets', exist_ok=True)
    torch.save(result, 'datasets/delta_to_dir.pt')
    return result

loss = torch.nn.SmoothL1Loss
accuracy_style = 'multiclass'

def main():
    import os
    import json
    import logging
    import logging.config
    import matplotlib.pyplot as plt
    import numpy as np

    folder = os.path.join('out', 'dtd-nonlins')
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
        __name__, # module
        FixedSweep.with_fixed_trials(  # Trial selector
            list((s,) for s in NONLINEARITIES),  # one variable
            3,  # trials per nonlinearity
        ),
        3,
        'fastest',
        folder
    )

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
