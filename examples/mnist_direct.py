"""Calls the trainer directly to train a model on mnist relatively quickly,
however this stores nothing useful after doing all that work. It only prints
out which epochs it completed.
"""
import torchluent
import ignite_simple
from ignite.engine import Events
import torchvision
import torch
import torch.utils.data as data

_module = 'examples.mnist_direct'

def _model():
    return (
        torchluent.FluentModule((1, 28, 28))
        .wrap(True)
        .conv2d(32, 5)
        .maxpool2d(3)
        .operator('LeakyReLU')
        .save_state()
        .flatten()
        .dense(64)
        .operator('Tanh')
        .save_state()
        .dense(10)
        .save_state()
        .build(with_stripped=True)
    )[1]

def _task(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_set, val_set, train_loader

loss = torch.nn.CrossEntropyLoss

def _log_epoch(tnr, state):
    print(f'Epoch Completed: {tnr.state.epoch}')

def main():
    """Trains a model on mnist"""
    ignite_simple.trainer.train(ignite_simple.trainer.TrainSettings(
        'classification',
        (_module, '_model', [], dict()),
        (_module, 'loss', [], dict()),
        (_module, '_task', [64], dict()),
        [(Events.EPOCH_COMPLETED, (_module, '_log_epoch', [], dict()))],
        1e-4,
        0.5,
        6,
        3
    ))

if __name__ == '__main__':
    main()
