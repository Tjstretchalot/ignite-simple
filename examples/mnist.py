"""Trains a model on mnist
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

def _dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss

def main(is_continuation, hparams):
    """Trains a model on mnist"""
    ignite_simple.train(
        (__name__, '_model', [], dict()),
        (__name__, '_dataset', [], dict()),
        (__name__, 'loss', [], dict()),
        folder='out/examples/mnist/current',
        hyperparameters=hparams,
        analysis='images',
        allow_later_analysis_up_to='video',
        accuracy_style='classification',
        trials=1,
        is_continuation=is_continuation,
        history_folder='out/examples/mnist/history',
        cores='all'
    )

def reanalyze():
    """Reanalyzes the existing trials, possibly under different analysis
    settings"""
    ignite_simple.analyze(
        (__name__, '_dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder='out/examples/mnist/current',
        settings='video',
        accuracy_style='classification',
        cores='all')

if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Simple model/dataset example')
    parser.add_argument('--no_continue', action='store_true',
                        help='Set is_continuation to False')
    parser.add_argument('--hparams', type=str, default='fast',
                        help='Which hyperparameter preset to use')
    parser.add_argument('--reanalyze', action='store_true',
                        help='Reanalyze instead of performing additional trials')
    args = parser.parse_args()

    if args.reanalyze:
        reanalyze()
    else:
        main(not args.no_continue, args.hparams)
