"""Utility functions which are used through ignite-simple but are short
enough not to warrant their own module.
"""

import torch.utils.data as data
import torch
import numpy as np
import typing
import importlib
import os
import sys

def noop(*args, **kwargs):
    """This function does nothing."""
    pass

def fix_imports(loader: typing.Tuple[str, str, tuple, dict]):
    """Returns the loader which represents the same callable as the given one,
    except potentially with the __main__ name replaced with the correct module
    name.

    :param loader: the loader whose imports might need to be cleaned

    :returns: the cleaned loader
    """
    if loader[0] != '__main__':
        return loader

    called_file = sys.argv[0]
    if called_file == '-c':
        raise ValueError('cannot fix __main__ when python invoked with -c')

    called_file = os.path.abspath(called_file)
    cwd = os.getcwd()

    if not called_file.startswith(cwd):
        raise ValueError('cannot fix __main__ when not called from a parent '
                         + 'directory. you should be called with something '
                         + 'like python -m examples.mnist_direct or '
                         + 'python examples/mnist_direct.py but never '
                         + 'python ../examples/mnist_direct.py')

    called_file = called_file[len(cwd):]

    called_file, _ = os.path.splitext(called_file)
    new_path = []
    cur_path, tail = os.path.split(called_file)
    new_path.append(tail)
    while cur_path:
        new_cur_path, tail = os.path.split(cur_path)
        if tail:
            new_path.append(tail)
        if new_cur_path == cur_path:
            break
        cur_path = new_cur_path
    new_path.reverse()

    correct_module = '.'.join(new_path)
    res = [correct_module]
    res.extend(loader[1:])
    return tuple(res)

def invoke(loader: typing.Tuple[str, str, tuple, dict]):
    """Invokes the callable which has the given name in the given module,
    using the given arguments and keyword arguments

    :param loader: (module, attrname, args, kwargs) - the callable to invoke

    :returns: the result of the callable
    """
    modulename, attrname, args, kwargs = loader

    module = importlib.import_module(modulename)
    return getattr(module, attrname)(*args, **kwargs)

def create_partial_loader(dset: data.Dataset, amt: int,
                          batch_size: int = 256) -> data.DataLoader:
    """Creates a dataloader which loads only a random subset of the
    specified length from the dataset, using the specified batch size.

    :param dset: the dataset to create a loader for a partial subset of

    :param amt: the number of items in the partial subset

    :returns: the described dataloader with a reasonable batch size
    """
    if amt == len(dset):
        return data.DataLoader(dset, batch_size=batch_size)

    inds = np.random.choice(len(dset), amt, replace=False)
    inds = torch.from_numpy(inds).long()
    return data.DataLoader(data.Subset(dset, inds),
                           batch_size=batch_size)

def task_loader(dataset_loader, batch_size, shuffle, drop_last):
    """Creates a task loader from a dataset loader.

    :param dataset_loader: the dataset loader (str, str, tuple, dict)
    :param batch_size: the batch size for the data loader
    :param shuffle: if the training dataset should be shuffled between epochs
    :param drop_last: if the last batch should be dropped if its not the
        same size as the rest. should only be used if shuffle is True
    """
    train_set, val_set = invoke(dataset_loader)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_set, val_set, train_loader

def split(full: data.Dataset,
          val_perc: float,
          filen: str = None) -> typing.Tuple[data.Dataset, data.Dataset]:
    """Splits the given dataset into two datasets, the first of which has
    (1 - val_perc) fraction of the data and the other has val_perc fraction
    of the data, distributed randomly.

    :param data.Dataset full: the entire dataset to split into two
    :param float val_perc: the amount to be broken away from full

    :returns: (train_set, val_set)
    """
    if filen is not None:
        _, ext = os.path.splitext(filen)
        if ext == '':
            filen += '.npz'
        elif ext != '.npz':
            raise ValueError(f'bad file extension: {filen} (should be .npz)')

    if filen is None or not os.path.exists(filen):
        n_held_out = int(len(full) * val_perc)
        held_out_inds_np = (
            np.random.choice(len(full), n_held_out, replace=False))

        if filen is not None:
            np.savez_compressed(filen, held_out_inds=held_out_inds_np)
    else:
        with np.load(filen) as infile:
            held_out_inds_np = infile['held_out_inds']

    held_out_inds = torch.from_numpy(held_out_inds_np).long()
    not_held_out = torch.arange(len(full))
    not_held_out[held_out_inds] = -1
    not_held_out = not_held_out[not_held_out != -1]

    return data.Subset(full, not_held_out), data.Subset(full, held_out_inds)
