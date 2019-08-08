"""This example trains a linear network to go from a delta
(dx, dy) to the relative onehot direction (left, up, right, down).
"""

import ignite_simple  # pylint: disable=unused-import
import torch
import ignite_simple.tuner
import ignite_simple.hyperparams
import ignite_simple.analarams
import ignite_simple.utils
from torchluent import FluentModule
import logging
import json
import psutil

def model():
    """Creates the model that should be trained"""
    return (
        FluentModule((2,))
        .dense(4)
        .build()
    )

def dataset(max_abs_val: int = 30):
    """Creates a dataset that has the given maximum absolute value
    for the deltas

    Args:
        max_abs_val (int, optional): maximum values for each component. Defaults to 30.
    """

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

    return ignite_simple.utils.split(dset, 0.1)

loss = torch.nn.SmoothL1Loss

_module = 'examples.delta_to_dir'

def main():
    """Finds the correct learning rate range and batch size"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    hparams = ignite_simple.hyperparams.fast()
    aparams = ignite_simple.analarams.video()

    ignite_simple.tuner.tune(
        (_module, 'model', tuple(), dict()),
        (_module, 'dataset', tuple(), dict()),
        (_module, 'loss', tuple(), dict()),
        'inv-loss',
        'out/examples/delta_to_dir',
        psutil.cpu_count(logical=False),
        hparams,
        aparams,
        logger
    )

if __name__ == '__main__':
    main()
