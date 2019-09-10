"""The goal of this example is to understand the effect of the number of hidden
nodes on a recurrent networks ability (with a readout) to match the dynamics of
a particular sequence which is generated by another network with the same
architecture but has a fixed number of hidden nodes.
"""
import torch
import torch.utils.data.dataset as dset
import os
import ignite_simple.utils as iutils
import math

STREAM_SIZE = 32
REFERENCE_SIZE = 128
NUM_SEQUENCES = 3000  # this is not much data; increase this to see the
                      # effect more clearly
HPARAMS = 'fastest'  # rnns aren't the most stable so consider increasing
                     # this to fast if having trouble
TRIALS_TO_FIND_HPARAMS = 3
TRIALS_WITH_FOUND_HPARAMS = 3
SEQ_LENGTHS = 10
SIZES_TO_CHECK = [8, 64, 128, 256]
# Alternatives: [1, 2, 4, 8, 16, 32, 64, 128, 256]
# list(range(8, 256+1, 8)) = [8, 16, 24, ..., 256]


class MyModel(torch.nn.Module):
    def __init__(self, hidden_nodes: int):
        super().__init__()
        self.rnn = torch.nn.RNN(STREAM_SIZE, hidden_nodes)
        self.readout = torch.nn.Linear(hidden_nodes, STREAM_SIZE)
        self.hidden_nodes = hidden_nodes

    def forward(self, inp):
        # inp has shape (batch_size, STREAM_SIZE)
        batch = inp.shape[0]
        inp = inp.reshape((1, batch, STREAM_SIZE))
        inp = torch.cat(
            [inp, torch.zeros((SEQ_LENGTHS - 1, inp.shape[1], inp.shape[2]))]
        )
        # inp has shape (seq_len, batch, input_size)
        hid, nxt = self.rnn(inp)
        # hid has shape (seq_len, batch, self.hidden_nodes)
        hid = hid.transpose(0, 2)
        # hid has shape (self.hidden_nodes, batch, seq_len)
        hid = hid.transpose(1, 2)
        # hid has shape (self.hidden_nodes, seq_len, batch)
        hid = hid.reshape(self.hidden_nodes, SEQ_LENGTHS * batch)
        # hid has shape (self.hidden_nodes, seq_len*batch)
        hid = hid.transpose(0, 1)
        # hid has shape (seq_len*batch, self.hidden_nodes)
        outs = self.readout(hid)
        # outs has shape (seq_len*batch, stream_size)
        outs = outs.transpose(0, 1)
        # outs has shape (stream_size, seq_len*batch)
        outs = outs.reshape(STREAM_SIZE, SEQ_LENGTHS, batch)
        # outs has shape (stream_size, seq_len, batch)
        outs = outs.transpose(0, 2)
        # outs has shape (batch, seq_len, stream_size)
        outs = outs.reshape(batch, SEQ_LENGTHS * STREAM_SIZE)
        # outs has shape (batch, seq_len*stream_size)
        return outs

model = MyModel

DATASET_FILE = os.path.join('datasets', 'seqmatch_hidden.pt')
def dataset(build=False):
    if not build and os.path.exists(DATASET_FILE):
        return torch.load(DATASET_FILE)

    # take a network, initialize it to be suuuper chaotic, then just run
    # some random data through it

    os.makedirs('datasets', exist_ok=True)

    mdl = model(REFERENCE_SIZE)

    # make the model really chaotic
    mdl.rnn.weight_hh_l0.data = (16 / math.sqrt(REFERENCE_SIZE)) * torch.randn((mdl.rnn.weight_hh_l0.shape))
    mdl.rnn.weight_hh_l0.data += torch.eye(REFERENCE_SIZE)

    starts = torch.rand((NUM_SEQUENCES, STREAM_SIZE))
    with torch.no_grad():
        seqs = mdl(starts)

    full = dset.TensorDataset(starts, seqs)
    tr, vl = iutils.split(full, 0.1)
    torch.save((tr, vl), DATASET_FILE)
    return tr, vl

loss = torch.nn.MSELoss
accuracy_style = 'inv-loss'

def main():
    import logging.config
    import ignite_simple.gen_sweep.sweeper as sweeper
    from ignite_simple.gen_sweep.param_selectors import FixedSweep

    folder = os.path.join('out', 'seqmatch_hidden')
    dataset()  # make sure this exists

    if os.path.exists('logging-gen.conf'):
        logging.config.fileConfig('logging-gen.conf')
    elif os.path.exists('logging.conf'):
        logging.config.fileConfig('logging.conf')
    else:
        print('No logging file found! Either cancel with Ctrl+C or press')
        print('enter to continue with no output.')
        input()

    sweeper.sweep(
        __name__, # module
        FixedSweep.with_fixed_trials(  # Trial selector
            tuple((s,) for s in SIZES_TO_CHECK),
            TRIALS_WITH_FOUND_HPARAMS
        ),
        TRIALS_TO_FIND_HPARAMS,
        HPARAMS,
        folder
    )

def replot():
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    folder = os.path.join('out', 'seqmatch_hidden')
    res_file = os.path.join(folder, 'points', 'funcres.pt')
    with open(res_file, 'rb') as infile:
        res = pickle.load(infile)

    # see readme in ignite_simple/gen_sweep for how res looks
    # or the documentation for sweeper.sweep
    xs = np.array([pt[0][0] for pt in res])
    ys = np.array([pt[7] for pt in res])
    # ys has shape (len(SIZES_TO_CHECK), TRIALS_WITH_FOUND_HPARAMS)

    fig, ax = plt.subplots()
    ax.set_title(f'Number of hidden to loss')
    ax.set_xlabel('Num. Hidden')
    ax.set_ylabel('Loss (Validation)')
    ax.plot(xs, ys)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'results.pdf'))

    plt.close(fig)


if __name__ == '__main__':
    if not os.path.exists(os.path.join('out', 'seqmatch_hidden')):
        main()
    replot()
