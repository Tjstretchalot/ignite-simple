"""Train a deep convolutional net on cifar10
"""

import torchluent
import ignite_simple
import ignite_simple.helper
import torchvision
import torch
import os
import ignite_simple.hyperparams as hyperparams

def model():
    return (
        torchluent.FluentModule((3, 32, 32))
        .conv2d(64, 4, padding=1, stride=2)
        .operator('ReLU')
        .operator('Dropout', 0.25)
        .conv2d(64, 2, padding=1, stride=2)
        .operator('ReLU')
        .operator('Dropout', 0.25)
        .conv2d(32, 3, padding=1, stride=3)
        .operator('ReLU')
        .operator('Dropout', 0.15)
        .flatten()
        .dense(128)
        .operator('Dropout', 0.25)
        .operator('Tanh')
        .dense(10)
        .build()
    )

def dataset(download=False):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(
        'datasets/cifar10', download=download, transform=transform)
    val_set = torchvision.datasets.CIFAR10(
        'datasets/cifar10', train=False, download=download,
        transform=transform)

    return train_set, val_set

loss = torch.nn.CrossEntropyLoss
accuracy_style = 'classification'

if __name__ == '__main__':
    dataset(True)
    hparams = hyperparams.slow()
    hparams.lr_sweep_len = 20.
    hparams.batch_pts = 3

    ignite_simple.train(
        (__name__, 'model', tuple(), dict()),
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder=os.path.join('out', 'examples', 'cifar10', 'current'),
        hyperparameters=hparams,
        analysis='images-min',
        allow_later_analysis_up_to='images-min',
        accuracy_style=accuracy_style,
        trials=1,
        is_continuation=False,
        history_folder=os.path.join('out', 'examples', 'cifar10', 'history'),
        cores='all',
        trials_strict=False
    )
