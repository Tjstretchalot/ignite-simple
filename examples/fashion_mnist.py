"""Trains a model on fashion mnist
"""
import torchluent
import ignite_simple
import torchvision
import torch
import logging.config

def _model():
    # From https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
    return (
        torchluent.FluentModule((1, 28, 28))
        .wrap(True)
        .conv2d(32, 5, 1, 2)
        .operator('ReLU')
        .maxpool2d(2, 2)
        .save_state()
        .conv2d(64, 5, 1, 2)
        .operator('ReLU')
        .maxpool2d(2, 2)
        .flatten()
        .save_state()
        .dense(1028)
        .operator('ReLU')
        .save_state()
        .operator('Dropout', 0.4)
        .dense(10)
        .save_state()
        .build(with_stripped=True)
    )

def _dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(
        'datasets/fashion_mnist', download=True, transform=transform)
    val_set = torchvision.datasets.FashionMNIST(
        'datasets/fashion_mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss

def main(is_continuation, hparams, cores):
    """Trains a model on fashion mnist"""
    ignite_simple.train(
        (__name__, '_model', [], dict()),
        (__name__, '_dataset', [], dict()),
        (__name__, 'loss', [], dict()),
        folder='out/examples/fashion_mnist/current',
        hyperparameters=hparams,
        analysis='images',
        allow_later_analysis_up_to='video',
        accuracy_style='classification',
        trials=1,
        is_continuation=is_continuation,
        history_folder='out/examples/fashion_mnist/history',
        cores=cores
    )

def reanalyze(cores):
    """Reanalyzes the existing trials, possibly under different analysis
    settings"""
    ignite_simple.analyze(
        (__name__, '_dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder='out/examples/fashion_mnist/current',
        settings='images',
        accuracy_style='classification',
        cores=cores)

if __name__ == '__main__':
    logging.config.fileConfig('logging.conf')

    import argparse
    parser = argparse.ArgumentParser(description='Simple model/dataset example')
    parser.add_argument('--no_continue', action='store_true',
                        help='Set is_continuation to False')
    parser.add_argument('--reanalyze', action='store_true',
                        help='Reanalyze instead of performing additional trials')
    parser.add_argument('--hparams', type=str, default='fast',
                        help='Which hyperparameter preset to use')
    parser.add_argument('--cores', type=int, default=-1,
                        help='number of cores to use')
    args = parser.parse_args()

    if args.reanalyze:
        reanalyze(args.cores if args.cores != -1 else 'all')
    else:
        main(not args.no_continue, args.hparams, args.cores if args.cores != -1 else 'all')
