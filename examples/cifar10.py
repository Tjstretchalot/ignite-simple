"""Train a deep convolutional net on cifar10
"""

import torchluent
import ignite_simple
import ignite_simple.helper
import torchvision
import torch

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
    ignite_simple.helper.handle(__name__)
