"""This example trains a linear network to go from a delta
(dx, dy) to the relative onehot direction (left, up, right, down). This is a
very simple task with a small dataset, which means it's an example that
can be run trivially on a laptop.
"""

import ignite_simple
import torch
from torchluent import FluentModule
import logging.config


class MyNonlin(torch.nn.Module):
    """The nonlinearity used by the model"""
    def forward(self, x):  # pylint: disable=arguments-differ
        r"""
        .. math::

            \frac{tanh(x)}{2} + 0.5

        """
        return (torch.tanh(x) / 2) + 0.5

def model():
    """Creates the model that should be trained"""
    return (
        FluentModule((2,))
        .wrap(True)
        .dense(4)
        .then(MyNonlin())
        .save_state()
        .build(with_stripped=True)
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

def main(is_continuation):
    """Finds the correct learning rate range and batch size"""
    ignite_simple.train(
        (__name__, 'model', tuple(), dict()),
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder='out/examples/delta_to_dir/current',
        hyperparameters='fast',
        analysis='images',
        allow_later_analysis_up_to='video',
        accuracy_style='multiclass',
        trials=1,
        is_continuation=is_continuation,
        history_folder='out/examples/delta_to_dir/history',
        cores='all'
    )

def reanalyze():
    """Reanalyzes the existing trials, possibly under different analysis
    settings"""
    ignite_simple.analyze(
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder='out/examples/delta_to_dir/current',
        settings='images',
        accuracy_style='multiclass',
        cores='all')

if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Simple model/dataset example')
    parser.add_argument('--no_continue', action='store_true',
                        help='Set is_continuation to False')
    parser.add_argument('--reanalyze', action='store_true',
                        help='Reanalyze instead of performing additional trials')
    args = parser.parse_args()

    if args.reanalyze:
        reanalyze()
    else:
        main(not args.no_continue)
