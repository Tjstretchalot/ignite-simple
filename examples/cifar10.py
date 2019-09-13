"""Train a model inspired by
https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

Further simplified.
"""

import torchluent
import ignite_simple
import ignite_simple.helper
import torchvision
import torch

def model():
    return (
        torchluent.FluentModule((3, 32, 32))
        .conv2d(32, 3, padding=1)
        .operator('PReLU')
        .conv2d(32, 3, padding=1)
        .operator('PReLU')
        .maxpool2d(2)
        .operator('Dropout', 0.2)
        .conv2d(64, 3, padding=1)
        .operator('PReLU')
        .conv2d(64, 3, padding=1)
        .operator('PReLU')
        .maxpool2d(2)
        .operator('Dropout', 0.3)
        .conv2d(128, 3, padding=1)
        .operator('PReLU')
        .conv2d(128, 3, padding=1)
        .operator('PReLU')
        .maxpool2d(2)
        .operator('Dropout', 0.4)
        .flatten()
        .dense(10)
        .build()
    )

def dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(
        'datasets/cifar10', download=True, transform=transform)
    val_set = torchvision.datasets.CIFAR10(
        'datasets/cifar10', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss
accuracy_style = 'classification'

if __name__ == '__main__':
    ignite_simple.helper.handle(__name__)
