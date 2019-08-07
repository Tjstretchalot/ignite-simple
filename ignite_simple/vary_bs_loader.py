"""A torch dataloader which varies the batch size between two amounts
over the course of a specified number of epochs of an underlying dataset. The
variation spends more time at lower batch sizes than at higher batch sizes, for
convenience of implementation and to account for the higher stochasticity at
lower batch sizes
"""

import torch

class BatchSizeVaryingDataLoader:
    """A dataloader which acts on an underlying dataset, varying the batch size
    linearly between two specified amounts over a given period of time. Note
    that this redefines one epoch to be the specified amount of time!

    :ivar data.Dataset dataset: the underlying dataset from which points and
        labels are being pulled
    :ivar int start_batch_size: the starting batch size
    :ivar int end_batch_size: the final batch size
    :ivar int epochs: the number of epochs over which the underlying dataset
        is iterated over
    :ivar iterator last_iter: the last real iterator that was created
    """
    def __init__(self, dataset, start_batch_size, end_batch_size, epochs):
        self.dataset = dataset
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.epochs = epochs
        self.last_iter = None
        self._len = None

    def __iter__(self):
        res = _BatchSizeVaryingDataLoaderIter(self, self._len)
        self._len = len(res)
        self.last_iter = res
        return res

    def dry_iter(self):
        """Creates a 'dry' iterator which does not actually produce anything
        but has the correct length and updates last_batch_size normally"""
        res = _BatchSizeVaryingDataLoaderIter(self, self._len, True)
        self._len = len(res)
        return res

    def __len__(self):
        if self._len is None:
            iter(self)
        return self._len

class _BatchSizeVaryingDataLoaderIter:
    def __init__(self, loader, len_=None, dry=False):
        self.dataset = loader.dataset
        self.start_batch_size = loader.start_batch_size
        self.end_batch_size = loader.end_batch_size
        self.epochs = loader.epochs
        self.last_batch_size = None
        self.dry = dry

        dset_len = len(self.dataset)
        self.batch_sizes = torch.linspace(
            self.start_batch_size,
            self.end_batch_size,
            dset_len * self.epochs
        ).long()

        self.position = 0
        if not len_:
            len_ = 1
            pos = int(self.batch_sizes[0]) if dset_len > 0 else 1
            while pos < dset_len * self.epochs:
                bsize = self.batch_sizes[pos]
                pos += bsize
                len_ += 1
            len_ -= 1

        self._len = len_

    def __next__(self):
        bsize = self.batch_sizes[self.position]
        dlen = len(self.dataset)
        if self.position + bsize >= dlen * self.epochs:
            raise StopIteration

        if not self.dry:
            points = []
            lbls = []

            rind = self.position % dlen
            for _ in range(bsize):
                pt, lbl = self.dataset[rind]
                points.append(pt)
                lbls.append(lbl)

                rind += 1
                if rind == dlen:
                    rind = 0
        else:
            points = [0]
            lbls = [0]

        self.position += bsize
        self.last_batch_size = bsize
        return self._collate(points), self._collate(lbls)

    def _collate(self, arr):
        if isinstance(arr[0], torch.Tensor):
            return torch.stack(arr, 0)
        return torch.tensor(arr)

    def __iter__(self):
        return self

    def __len__(self):
        return self._len
