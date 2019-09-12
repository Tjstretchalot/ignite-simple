"""Investigates the effect of model initialization on performance for a delayed
classification task for an RNN. The task is to classify points which come from
clusters.
"""
import ignite_simple.utils
import ignite_simple
import ignite_simple.hyperparams as hyperparameters
import torch
import torch.utils.data as tdata
import numpy as np
import math
import os
import typing
from scipy.spatial.distance import cdist

INPUT_DIM = 2
CLUSTER_STD = 0.005
N_CLUSTERS = 60
POINTS_PER_CLUST = 50000 // N_CLUSTERS

# ensures 99.9999% chance that points can be correctly identified
CLUSTER_MIN_SEP = CLUSTER_STD * (0.59943537 * ((INPUT_DIM - 1) ** 0.57832581) + 4.891638480163717)

HIDDEN_SIZE = 128
SEQ_LENGTH = 10  # first is input, last+1 is readout
OUTPUT_DIM = 2
DATASET_FILE = os.path.join('datasets', 'advanced', 'gaussian_blobs_delayed_class')

FOLDER = os.path.join('out', 'advanced', 'gaussian_blobs_delayed_class', 'sweep')
FOLDER_SINGLE = os.path.join('out', 'advanced', 'gaussian_blobs_delayed_class', 'single')
TRIALS_TO_FIND_HPARAMS = 3
TRIALS_WITH_FOUND_HPARAMS = 3
HPARAMS = 'fastest'
G_VALUES = (0, 0.2, 0.25, 0.3, 0.35, 0.4)

def gaussian_blobs(means: torch.tensor, std: float, num_samples_per_blob: int,
                   odim: int) -> typing.Tuple[torch.tensor, torch.tensor]:
    """Produces labeled points which come from gaussian blobs with the given
    positions.

    :param means: centers for the blobs
    :param std: std for all the blobs (one std for all blobs)
    :param num_samples_per_blob: the number of samples in each blob
    :return: points and their corresponding cluster index, unshuffled
    """
    number_of_blobs = means.shape[0]
    dim = means.shape[1]

    blobs = []
    indices = []
    for i, mean in enumerate(means):
        blob = torch.randn(num_samples_per_blob, dim) * std + mean
        idxs = torch.zeros(num_samples_per_blob, dtype=torch.int32) + i
        blobs.append(blob)
        indices.append(idxs)

    points = torch.cat(blobs, dim=0)
    indices = torch.cat(indices, dim=0)

    return points.float(), indices

def draw_centers(dim: int, n_clusters: int, min_sep: float) -> torch.tensor:
    """Determines where the centers of clusters should go. Points are drawn
    uniformly at random from a hypercube with side length 2 centered at the
    origin. Points are rejected if they are within CLUSTER_MIN_SEP of each
    other.
    """
    num_rejects = 0
    result = np.random.uniform(-1, 1, (1, dim))

    at_a_time = min(N_CLUSTERS - 1, 64)

    while result.shape[0] < n_clusters:
        pt = np.random.uniform(-1, 1, (at_a_time, dim))
        nearest = np.min(cdist(result, pt))
        if nearest >= min_sep:
            result = np.concatenate((result, pt))
            at_a_time = min(at_a_time, n_clusters - result.shape[0])
        else:
            if at_a_time == 1:
                num_rejects += 1
                if num_rejects > 10000:
                    raise ValueError('rejected too many samples')
            else:
                at_a_time = max(at_a_time - 1, 1)

    return torch.from_numpy(result).float()


class MyModel(torch.nn.Module):
    """A simple RNN with a linear readout. Initialized with a sompolinsky
    approximation for fixed gain.

    :ivar torch.nn.RNN rnn: what does the bulk of the work
    :ivar torch.nn.Linear readout: the final readout
    :ivar float gain: the gain this model was initialized with
    :ivar bool return_hidden: True to change the result from forward to
        (tensor[batch, OUTPUT_DIM], list(tensor[batch, HIDDEN_SIZE]))
        for interpetation
    """

    def __init__(self, gain: float):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                                nonlinearity='tanh')
        self.readout = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_DIM)
        self.gain = gain
        self.return_hidden = False

        std = gain / math.sqrt(HIDDEN_SIZE)

        rnn_w = self.rnn.weight_hh_l0.data
        rnn_w[:] = 0

        torch.eye(HIDDEN_SIZE, out=rnn_w)
        rnn_w += torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * std

    def shallow_with_hidden(self) -> 'MyModel':
        """Returns a shallow copy of this model with hidden results on"""
        res = MyModel(self.gain)
        res.rnn = self.rnn
        res.readout = self.readout
        res.gain = self.gain
        res.return_hidden = True
        return res

    def forward(self, inp):
        # inp has shape (batch, INPUT_DIM)
        batch = inp.shape[0]
        inp = inp.reshape((1, batch, INPUT_DIM))
        inp = torch.cat(
            [inp, torch.zeros((SEQ_LENGTH - 1, inp.shape[1], inp.shape[2]))]
        )
        # inp has shape (SEQ_LENGTH, batch, INPUT_DIM)
        hid, nxt = self.rnn(inp)
        # hid has shape (SEQ_LENGTH, batch, HIDDEN_SIZE)
        # nxt has shape (1, batch, HIDDEN_SIZE)
        nxt = nxt.reshape(batch, HIDDEN_SIZE)
        # nxt has shape (batch, HIDDEN_SIZE)
        out = self.readout(nxt)
        # out has shape (batch, OUTPUT_DIM)
        if self.return_hidden:
            return out, list(hid.detach())
        return out

def model(gain: float):
    mdl = MyModel(gain)
    return mdl.shallow_with_hidden(), mdl

def dataset(build=False):
    if not build and os.path.exists(DATASET_FILE):
        return torch.load(DATASET_FILE)

    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)

    centers = draw_centers(INPUT_DIM, N_CLUSTERS, CLUSTER_MIN_SEP)
    pts, clust_inds = gaussian_blobs(centers, CLUSTER_STD, POINTS_PER_CLUST,
                                     OUTPUT_DIM)

    clust_lbls = torch.randint(0, OUTPUT_DIM, (N_CLUSTERS,))
    targets = torch.zeros((N_CLUSTERS*POINTS_PER_CLUST, OUTPUT_DIM))

    for clust_ind in range(N_CLUSTERS):
        targets[clust_inds == clust_ind, clust_lbls[clust_ind]] = 1

    full = tdata.TensorDataset(pts, targets)
    tr, val = ignite_simple.utils.split(full, 0.1)
    torch.save((tr, val), DATASET_FILE)
    return tr, val

accuracy_style = 'classification'
loss = torch.nn.MSELoss

def main():
    import logging.config
    import ignite_simple.gen_sweep.sweeper as sweeper
    from ignite_simple.gen_sweep.param_selectors import FixedSweep

    dataset(True)

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
            tuple((s,) for s in G_VALUES),
            TRIALS_WITH_FOUND_HPARAMS
        ),
        TRIALS_TO_FIND_HPARAMS,
        HPARAMS,
        FOLDER
    )

def replot():
    import matplotlib.pyplot as plt
    import pickle

    res_file = os.path.join(FOLDER, 'points', 'funcres.pt')
    with open(res_file, 'rb') as infile:
        res = pickle.load(infile)

    xs = np.array([pt[0][0] for pt in res])
    ys = np.array([pt[6].mean() for pt in res])

    fig, ax = plt.subplots()
    ax.set_title('G vs Mean Accuracy')
    ax.set_xlabel('G')
    ax.set_ylabel('Accuracy (%) (Validation)')
    ax.plot(xs, ys)

    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, 'results.pdf'))

    plt.close(fig)

def single_gain(gain: float):
    """Looks carefully at the models training for a single gain."""
    import logging.config
    if os.path.exists('logging.conf'):
        logging.config.fileConfig('logging.conf')

    hparams = hyperparameters.fastest()
    hparams.lr_min_inits = 16 # finding learning rate for RNNs is hard

    dataset(True)
    ignite_simple.train(
        (__name__, 'model', (float(gain),), dict()),
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder=os.path.join(FOLDER_SINGLE, str(gain)),
        hyperparameters='fastest',
        analysis='images-min',
        allow_later_analysis_up_to='videos',
        accuracy_style=accuracy_style,
        trials=1,
        is_continuation=True,
        history_folder=os.path.join(FOLDER_SINGLE, 'history'),
        cores='all',
        trials_strict=False
    )


if __name__ == '__main__':
    # if not os.path.exists(FOLDER):
    #     main()
    # replot()

    single_gain(0.4)
