"""Utility functions for creating subsets of dataloaders
"""

import torch.utils.data as data
import torch
import numpy as np
import typing

def split(full: data.Dataset,
          val_perc: float) -> typing.Tuple[data.Dataset, data.Dataset]:
    """Splits the given dataset into two datasets, the first of which has
    (1 - val_perc) fraction of the data and the other has val_perc fraction
    of the data, distributed randomly.

    :param data.Dataset full: the entire dataset to split into two
    :param float val_perc: the amount to be broken away from full

    :returns: (train_set, val_set)
    """
    n_held_out = int(len(full) * val_perc)

    held_out_inds = torch.from_numpy(
        np.random.choice(len(full), n_held_out, replace=False)).long()
    not_held_out = torch.arange(len(full))
    not_held_out[held_out_inds] = -1
    not_held_out = not_held_out[not_held_out != -1]

    return data.Subset(full, not_held_out), data.Subset(full, held_out_inds)
