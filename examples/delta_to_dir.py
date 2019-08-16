"""This example trains a linear network to go from a delta
(dx, dy) to the relative onehot direction (left, up, right, down). This is a
very simple task with a small dataset, which means it's an example that
can be run trivially on a laptop.
"""

import ignite_simple
import ignite_simple.helper
import torch
from torchluent import FluentModule
import os

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
    torch.save(result, 'datasets/delta_to_dir.pt')
    return result

loss = torch.nn.SmoothL1Loss
accuracy_style = 'multiclass'

if __name__ == '__main__':
    ignite_simple.helper.handle(__name__)
