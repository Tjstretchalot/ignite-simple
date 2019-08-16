"""Trains a model on fashion mnist. This uses the tensorflow example network
for mnist, which gets 91.6% validation performance. Using the automated "slow"
hyperparameter preset, the result was: On the validation set, after training,
the highest performance was 0.9281 and the lowest loss was 0.2237575893163681,
while on average performance was 0.9248333333333335 ± 0.0015101508386765761 and
loss was 0.22723734169205026 ± 0.0025451971519325754.
"""
import torchluent
import ignite_simple
import ignite_simple.helper
import torchvision
import torch

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
accuracy_style = 'classification'

if __name__ == '__main__':
    ignite_simple.helper.handle(__name__)
