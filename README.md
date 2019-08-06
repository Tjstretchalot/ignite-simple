# Ignite Simple

This module provides the necessary functionality for rapidly prototyping
machine learning models on conventional datasets.

## Usage

You must explicitly specify the training set, validation set, and the loss.
Then a preset is applied for how to tune hyperparameters and the amount of
information to gather and export.

Although not required, it is recommended use
[torchluent](http://github.com/tjstretchalot/torchluent) to simplify the model
creation process. This package accepts models with tensor outputs or tuples of
the form (tensor, list of tensors), where the list of tensors contains
relevants snapshots of the data as it was transformed by the network.

This package supports repeating trials with the final selected hyperparameters,
either by specifying the number of repeats with `trials` or by simply
setting `is_continuation` to `True` and calling `train` with the same output
directory, or both.

[API Reference](https://tjstretchalot.github.io/ignite-simple/index.html)

```py
import torchluent
import ignite_simple
import torchvision

def main():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)

    model = (
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
        .build()
    )

    loss = torch.nn.CrossEntropyLoss()

    ignite_simple.train(model, train_set, val_set, loss, folder='out',
                        hyperparameters='fast', analysis='images',
                        allow_later_analysis_up_to='video',
                        trials=1, is_continuation=False,
                        history_folder='history', cores='all')

if __name__ == '__main__':
    main()
```

## Continuations

In the above example, by changing `is_continuation` to `True`, the file may be
invoked multiple times. With `is_continuation=True`, the hyperparameters will
be reused from the first run and the model and statistics will be saved
alongside (not overwriting) the existing ones. Furthermore, additional plots
(averaged accuracy, averaged loss, etc) will be available. With
`is_continuation=False` the output folder will be archived and moved into
the history folder with the current timestamp as its name prior to starting
the run.

## Validation sets

If your dataset is not already broken up into training and validation data, the
validation dataset can be replaced with a float in (0, 1) to have a random
subset of the training set held out and used as validation dataset. The number
of samples is the specified % of the overall data (i.e. 0.1 for 10% of overall
data held out for validation).

## Labels and accuracy measures

This package chooses an accuracy measure based on the shape and datatype of
the labels. If the labels are integers, it is assumed that the output of
the network is a one-hot encoding of the label. If the labels are a tensor
of integers, they are assumed to be one-hot multiclass labels. In all
other cases, accuracy is not measured and inverse loss is used as a proxy.

For multi-class labels, the threshold may be specified with the
`accuracy_threshold` keyword argument to train

## Automatic hyperparameter tuning

Valid presets are `fastest`, `fast`, `slow`, and `slowest`.

The learning rate and batch size are automatically tuned since they can
dramatically effect model performance. The methodology is inspired by
[Cyclical Learning Rates for Training Neural Networks, 2017](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7926641).

For tuning the learning rate, a reasonable cycle size is used and the process
is analagous to that described by the paper.

The batch size is found in a similar way - vary the batch size upward linearly
over a few epochs. The range of batch sizes where the accuracy increased is
found, and then batch sizes are randomly drawn from that range and tested for
a few epochs. The batch size with the highest accuracy at the end of the short
test is used.

The hyperparameter presets correspond roughly to:
    - How many trials to average tests over (i.e., for learning rate it can be
    beneficial to find the range via an average of several runs rather than the
    very noisy output of a single run)
    - How many batch sizes are attempted within the reasonable range
    - How many times to repeat the process for checking for batch size and
    learning rate interactions.

It is intended that the preset 'fast' is used while making rapid adjustments
and then 'slow' used occassionally as a sanity check. Then 'slowest' can be
used if you don't require any additional features and want to use this package
for a final model.

## Automatic analysis

Valid presets are `none`, `text`, `images`, `animations-draft`, `animations`,
`video-draft`, and `video`.

This package is capable of producing some explanation about how the model was
trained and some information about it's solution. The analysis includes:

- An explanation of the learning rules
- All relevant hyperparameters for the network and brief explanations of each
- Where relevant, how hyperparameters were selected
- Network representation in 3d principal-component-space

The analysis can be provided in text form, all the previous
plus image references, all the previous plus animation references, or all
the previous plus a video guide. The `-draft` settings produce lower-quality
(FPS and resolution) versions that are somewhat faster to generate.

Analysis can be performed after-the-fact assuming that sufficient data was
collected (which is specified in the `allow_later_analysis_up_to` parameter).
The following snippet performs video analysis on the first example without
repeating training:

```py
import ignite_simple

def main():
    ignite_simple.reanalyze(folder='out', analysis='video')

if __name__ == '__main__':
    main()
```

Example output with the video preset:

TODO coming soon

Note that reanalysis does not reproduce unless it believes the result would be
different from that which exists on the file system. The analysis output is
in the `analysis` subdirectory of the output folder, and then in a folder
indexed by the number of models trained in the current hyperparameter settings.

## Implementation details

This package trains with SGD on a linear cyclical learning rate rule for a
fixed number of epochs. The batch size is fixed throughout training, and the
validation dataset is not used during hyperparameter selection nor model
training, however it is measured and reported for analysis.
