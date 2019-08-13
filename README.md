# Ignite Simple

This module provides the necessary functionality for rapidly prototyping
machine learning models on conventional datasets.

## Usage

You must explicitly specify the model, training set, validation set, and the
loss. Then a preset is applied for how to tune hyperparameters and the amount
of information to gather and export.

Although not required, it is recommended use
[torchluent](http://github.com/tjstretchalot/torchluent) to simplify the model
creation process. This package accepts a single model with tensor outputs or
a tuple of two models where the first is the stripped model (returns only
a single tensor) and the second model has the same underlying parameters but
return values are of the form (tensor, list of tensors), where the list of
tensors contains relevants snapshots of the data as it was transformed by the
network.

This package supports repeating trials with the final selected hyperparameters,
either by specifying the number of repeats with `trials` or by simply
setting `is_continuation` to `True` and calling `train` with the same output
directory, or both.

[API Reference](https://tjstretchalot.github.io/ignite-simple/index.html)

```py
import torchluent
import ignite_simple
import torch
import torchvision
import logging.config

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
        .build()
    )

def dataset():
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        'datasets/mnist', download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(
        'datasets/mnist', train=False, download=True, transform=transform)
    return train_set, val_set

loss = torch.nn.CrossEntropyLoss

def main():
    # a reasonable logging.conf is in this repository to get you started.
    # you won't see any stdout without a logging config!
    logging.config.fileConfig('logging.conf')
    ignite_simple.train((__name__, 'model', tuple(), dict()),
                        (__name__, 'dataset', tuple(), dict()),
                        (__name__, 'loss', tuple(), dict()),
                        folder='out', hyperparameters='fast',
                        analysis='images', allow_later_analysis_up_to='video',
                        accuracy_style='classification',
                        trials=1, is_continuation=False,
                        history_folder='history', cores='all')

if __name__ == '__main__':
    main()
```

## Continuations and trials

In the above example, by changing `is_continuation` to `True`, the file may be
invoked multiple times. With `is_continuation=True`, the hyperparameters will
be reused from the first run and the model and statistics will be saved
alongside (not overwriting) the existing ones. Furthermore, additional plots
(averaged accuracy, averaged loss, etc) will be available. With
`is_continuation=False` the output folder will be archived and moved into
the history folder with the current timestamp as its name prior to starting
the run.

Note that trials is treated as the *minimum* number of trials to perform.
This will attempt to use all available cores (i.e., the number specified in
cores), which may mean multiple trials can be run in parallel without any
significant difference in runtime. This can be suppressed with the parameter
`trials_strict=True`.

## Validation sets

For automatic dataset splitting into training and validiation, one can use
`ignite_simple.utils.split` as follows:

```py
import ignite_simple.utils
import torch.utils.data as data

full: data.Dataset  # dataset to split
val_perc: float = 0.1  # perc in the validation set


train_set, val_set = ignite_simple.utils.split(
    full, val_perc, filen='mydataset/train_val_split')
```

The split is random and stored in the given file (extensionless is recommended,
in which the appropriate extension will be added). If the file already exists,
this returns the split stored in the specified file. This makes it easier to
verify the training and validation accuracy after the fact and simplifies
comparisons of models on the same dataset.

## Accuracy style

Valid values are `classification`, `multiclass`, and `inv-loss`. Classification
is for MNIST-style labels (labels are one-hot and the output of the network
is a one-hot encoding of the label). Multi-class is for when the labels are
one-hot encoded class labels extended to potentially multiple ones. `inv-loss`
uses 1/loss as the performance metric instead of accuracy.

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
repeating training, assuming its in the same file:

```py
def reanalyze():
    ignite_simple.analyze(
        (__name__, 'dataset', tuple(), dict()),
        (__name__, 'loss', tuple(), dict()),
        folder='out',
        settings='video',
        accuracy_style='classification',
        cores='all')
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
