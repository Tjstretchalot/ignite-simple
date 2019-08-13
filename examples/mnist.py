"""Calls the trainer directly to train a model on mnist relatively quickly,
however this stores nothing useful after doing all that work. It only prints
out which epochs it completed.
"""
import torchluent
import ignite_simple
import torchvision
import torch
import logging.config

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
    )

def _dataset(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss

def main():
    """Trains a model on mnist"""
    ignite_simple.train(
        (__name__, '_model', [], dict()),
        (__name__, '_dataset', [64], dict()),
        (__name__, 'loss', [], dict()),
        folder='out/examples/mnist/current',
        hyperparameters='fast',
        analysis='video',
        allow_later_analysis_up_to='video',
        accuracy_style='classification',
        trials=1,
        is_continuation=False,
        history_folder='out/examples/mnist/history',
        cores='all'
    )

if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')
    main()
