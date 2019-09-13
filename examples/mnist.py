"""Trains a model on mnist

Results on a 12-core running with fast hparams preset:

Training was performed using minibatched SGD without momentum or dampening.
Learning rate oscillated linearly, starting at 0.0004260232373932849 and
increasing to 0.13309664830409554 at the start of epoch 2 and then decreasing
back to 0.0004260232373932849 at the start of cycle 4. The batch size was fixed
to 51. A total of 12 trials were performed. On the training set, after
training, the highest performance was 0.9993 and the lowest loss was
0.006737891340255737, while on average performance was
0.9986416666666665±0.0003040239391159081 and loss was
0.008421249224742253±0.0011044572749568308. On the validation set, after
training, the highest performance was 0.9912 and the lowest loss was
0.026645280361175538, while on average performance was
0.9907333333333331±0.00035668224265054715 and loss was
0.027536519054571783±0.0004289494679617814.

Video: https://youtu.be/hFw8XM8nrqM
"""
import torchluent
import ignite_simple
import ignite_simple.helper
import torchvision
import torch

def model():
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

def dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss
accuracy_style = 'classification'

if __name__ == '__main__':
    ignite_simple.helper.handle(__name__)
