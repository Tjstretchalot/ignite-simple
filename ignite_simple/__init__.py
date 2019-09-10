"""Loads the major code for this module and performs necessary initialization"""

import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:  # pylint: disable=bare-except  # noqa: E722
    pass

__all__ = ['train', 'analyze']

import os

html_path = os.path.join(os.path.dirname(__file__), '../html')

if not os.path.exists(html_path):
    raise Exception(
        'ignite_simple is not installed correctly! '
        + f'It expects {html_path} (found relative to this file) '
        + 'to exist, like in the repo.  '
        + 'Uninstall, download the repo, and install with the existing option'
        + '(-e) in pip')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['GOTOBLAS_MAIN_FREE'] = '1'

import torch
torch.set_num_threads(1)

import time
import random
torch.manual_seed(int(time.time() * 1000))
torch.randint(1, 100, (1000,))
torch.manual_seed(int(torch.randint(1, 2**16, (1,)).item() * time.time()))

torch.randint(1, 100, (1000,))
torch.randint(1, 100, (int(torch.randint(1, 100, (1,)).item()),))

random.seed()

import numpy as np
np.seterr('raise')
np.random.seed()

from ignite_simple.model_manager import train
from ignite_simple.analysis import analyze
