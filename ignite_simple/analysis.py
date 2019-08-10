"""This module is tasked with generating analysis text, images, animations,
and videos using only the output from the model manager, the dataset, the
loss metric, and the accuracy style.
"""
import typing
import logging
import ignite_simple  # pylint: disable=unused-import
import ignite_simple.utils as utils
import ignite_simple.figure_utils as futils
import ignite_simple.analarams as aparams
import ignite_simple.dispatcher as dispatcher
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import os
import psutil

def _rawplot(infile, title, xlab, ylab, x_varname, x_slice, y_varname, y_slice,
             outfile_wo_ext, reduction='none'):
    """Performs a single plot of the y-axis vs x-axis by loading the given
    numpy file, fetching the arrays by the variable names from the file,
    slicing it as specified, and then plotting.

    :param str infile: the file to load the numpy data frame
    :param str title: the title of the plot
    :param str xlab: the label for the x-axis
    :param str ylab: the label for the y-axis
    :param str x_varname: the name for the array to get the x-data from
        within the numpy file at infile
    :param slice x_slice: the slice or tuple of slices for the x-data.
        the x-data will be squeezed as much as possible after slicing
    :param str y_varname: the name for the array to get the y-data from
        within the numpy file at infile
    :param slice y_slice: the slice or tuple of slices for the y-data.
        the y-data will be squeezed as much as possible after slicing
    :param str outfile_wo_ext: the path to the outfile without an extension
    :param str reduction: how we handle extra dimensions. one of the following:

        * `none`

            no reduction is performed

        * `each`

            each of the first dimension corresponds to one graph

        * `mean`

            take the mean over the first dimension and plot without errorbars

        * `mean_with_errorbars`

            take the mean over the first dimension and plot with errorbars from
            standard deviation

        * `mean_with_fillbtwn`

            take the mean over the first dimension and plot with shaded error
            region


    """
    with np.load(infile) as nin:
        xs = nin[x_varname][x_slice]
        ys = nin[y_varname][y_slice]

    new_shape = list(i for i in ys.shape if i != 1)
    ys = ys.reshape(new_shape)

    new_shape_x = list(i for i in xs.shape if i != 1)
    xs = xs.reshape(new_shape_x)
    if len(new_shape_x) != 1:
        xs = xs[0] # take first

    if len(new_shape) > 1:
        if reduction == 'mean':
            ys = ys.mean(0)
        elif reduction in ('mean_with_errorbars', 'mean_with_fillbtwn'):
            stds = ys.std(0)
            means = ys.mean(0)

            errs = 1.96*stds
            errs_low = ys - errs
            errs_high = ys + errs
            ys = means
        elif reduction == 'each':
            pass
        else:
            raise ValueError(
                f'cannot reduce shape {new_shape} with {reduction}')
    else:
        reduction = 'none'

    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_figwidth(19.2)
    fig.set_figheight(10.8)
    fig.set_dpi(100)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if reduction in ('none', 'mean', 'mean_with_fillbtwn'):
        ax.plot(xs, ys)

        if reduction == 'mean_with_fillbtwn':
            ax.fill_between(xs, errs_low, errs_high)
    elif reduction == 'mean_with_errorbars':
        ax.errorbar(xs, ys, errs)
    elif reduction == 'each':
        for y in ys:
            ax.plot(xs, y)
    else:
        raise ValueError(f'unknown reduction {reduction}')

    futils.set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(48)
    ax.yaxis.label.set_size(48)
    fig.savefig(outfile_wo_ext + '_1920x1080.png', dpi=100)
    futils.set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    fig.savefig(outfile_wo_ext + '_19.2x10.8.pdf', dpi=300, transparent=True)

    fig.set_figwidth(7.25)  # paper width
    fig.set_figheight(4.08)  # 56.25% width

    futils.set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    fig.savefig(outfile_wo_ext + '_725x408.png', dpi=100)
    futils.set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    fig.savefig(outfile_wo_ext + '_7.25x4.08.pdf', dpi=300, transparent=True)

    fig.set_figwidth(3.54)  # column width
    fig.set_figheight(1.99)  # 56.25% width

    futils.set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    fig.savefig(outfile_wo_ext + '_354x199.png', dpi=100)
    futils.set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.savefig(outfile_wo_ext + '_3.54x1.99.pdf', dpi=300, transparent=True)

    fig.set_figwidth(1.73)  # half column width
    fig.set_figheight(0.972)  # 56.25% width

    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)

    futils.set_title(fig, ax, title, True)
    fig.savefig(outfile_wo_ext + '_173x97.png', dpi=100)
    futils.set_title(fig, ax, title, False)
    fig.savefig(outfile_wo_ext + '_1.73x97.pdf', dpi=300, transparent=True)

    fig.set_figwidth(1.73)  # half column width
    fig.set_figheight(1.73)  # square

    futils.set_title(fig, ax, title, True)
    fig.savefig(outfile_wo_ext + '_173x173.png', dpi=100)
    futils.set_title(fig, ax, title, False)
    fig.savefig(outfile_wo_ext + '_1.73x1.73.pdf', dpi=300, transparent=True)

    plt.close(fig)


def analyze(dataset_loader: typing.Tuple[str, str, tuple, dict],
            loss_loader: typing.Tuple[str, str, tuple, dict],
            folder: str,
            settings: typing.Union[str, aparams.AnalysisSettings],
            accuracy_style: str, cores: int):
    """Performs the requested analysis of the given folder, which is assumed to
    have been generated from model_manager.train. The output folder structure
    is as follows:

    .. code:: none

        folder/
            analysis/
                hparams/
                    lr/
                        i/  (where i=0,1,...)
                            lr_vs_perf.(img)
                            lr_vs_perf_deriv.(img)
                            lr_vs_smoothed_perf_deriv.(img)

                        lr_vs_perf_each.(img)
                        lr_vs_perf_mean.(img)
                        lr_vs_perf_errorbars.(img)
                        lr_vs_perf_fillbtwn.(img)
                            errorbars like https://stackoverflow.com/a/13157955
                        lr_vs_perf_deriv_each.(img)
                        lr_vs_perf_deriv_mean.(img)
                        lr_vs_perf_deriv_errorbars.(img)
                        lr_vs_perf_deriv_fillbtwn.(img)
                        lr_vs_smoothed_perf_deriv_each.(img)
                        lr_vs_smoothed_perf_deriv_mean.(img)
                        lr_vs_smoothed_perf_deriv_errorbars.(img)
                        lr_vs_smoothed_perf_deriv_fillbtwn.(img)
                        lr_vs_smoothed_perf_deriv_range.(img)
                    batch/
                        i/ (where i=0,1,...)
                            batch_vs_perf.(img)
                            batch_vs_perf_deriv.(img)
                            batch_vs_smoothed_perf_deriv.(img)

                        batch_vs_perf_each.(img)
                        batch_vs_perf_mean.(img)
                        batch_vs_perf_errorbars.(img)
                        batch_vs_perf_fillbtwn.(img)
                        batch_vs_perf_deriv_each.(img)
                        batch_vs_perf_deriv_mean.(img)
                        batch_vs_perf_deriv_errorbars.(img)
                        batch_vs_perf_deriv_fillbtwn.(img)
                        batch_vs_smoothed_perf_deriv_each.(img)
                        batch_vs_smoothed_perf_deriv_mean.(img)
                        batch_vs_smoothed_perf_deriv_errorbars.(img)
                        batch_vs_smoothed_perf_deriv_fillbtwn.(img)
                        batch_vs_smoothed_perf_deriv_range.(img)

                    TODO videos & animations
                trials/
                    i/  (where i=0,1,...)
                        epoch_vs_loss.(img)
                            Only produced if throughtime.npz is available for
                            the trial and settings.training_metric_images is
                            set
                        epoch_vs_perf.(img)
                            Only produced if throughtime.npz is available for
                            the trial and settings.training_metric_images is
                            set. The axis title and figure title depend on the
                            accuracy style

                    TODO more summary of trials
                    TODO videos & animations

    :param dataset_loader: the module and corresponding attribute that gives a
        training and validation dataset when invoked with the specified
        arguments and keyword arguments.

    :param loss_loader: the module and corresponding attribute that gives a
        loss nn.Module when invoked with the specified arguments and keyword
        arguments

    :param folder: where the model manager saved the trials to analyze

    :param settings: the settings for analysis, or the name of the preset to
        use. Common preset names are `none`, `text`, `images`, `animations`,
        and `videos`. For the full list, see the ignite_simple.analarams
        module.

    :param accuracy_style: how performance was calculated. one of
        'classification', 'multiclass', and 'inv-loss'. See train for
        details.

    :param cores: the number of physical cores this can assume are available
        to speed up analysis.
    """
    if cores == 'all':
        cores = psutil.cpu_count(logical=False)

    dataset_loader = utils.fix_imports(dataset_loader)
    loss_loader = utils.fix_imports(loss_loader)

    settings = aparams.get_settings(settings)

    logger = logging.getLogger(__name__)

    logger.info('Analyzing %s...', folder)

    perf_name = (  # TODO
        'Accuracy' if accuracy_style in ('classification, multiclass')
        else 'Inverse-Loss'
    )

    tasks = []
    if settings.hparam_selection_specific_imgs:
        lr_folder = os.path.join(folder, 'analysis', 'hparams', 'lr')
        batch_folder = os.path.join(folder, 'analysis', 'hparams', 'batch')
        for suffix in ('', '2'):
            source = os.path.join(folder, 'hparams', f'lr_vs_perf{suffix}.npz')
            if not os.path.exists(source):
                continue

            out_folder = lr_folder + suffix
            with np.load(source) as infile:
                num_trials = infile['perfs'].shape[0]

            for trial in range(num_trials):
                real_out = os.path.join(out_folder, str(trial))
                os.makedirs(real_out, exist_ok=True)

                have_imgs = os.path.exists(
                    os.path.join(real_out, 'lr_vs_perf_1920x1080.png'))
                if have_imgs:
                    continue

                tasks.extend([
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Learning Rate vs Inverse Loss',
                            'Learning Rate',
                            '1/loss',
                            'lrs',
                            slice(None),
                            'perfs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'lr_vs_perf')
                        ),
                        dict(),
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Learning Rate vs Inverse Loss Deriv.',
                            'Learning Rate',
                            '1/loss Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'perf_derivs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'lr_vs_perf_deriv')
                        ),
                        dict(),
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Learning Rate vs Inverse Loss Deriv. (Smoothed)',
                            'Learning Rate',
                            '1/loss Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'smoothed_perf_derivs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'lr_vs_smoothed_perf_deriv')
                        ),
                        dict(),
                        1
                    ),
                ])

            source = os.path.join(folder, 'hparams', 'bs_vs_perf.npz')
            out_folder = batch_folder
            with np.load(source) as infile:
                num_trials = infile['perfs'].shape[0]

            for trial in range(num_trials):
                real_out = os.path.join(out_folder, str(trial))
                os.makedirs(real_out, exist_ok=True)

                have_imgs = os.path.exists(
                    os.path.join(real_out, 'batch_vs_perf_1920x1080.png'))
                if have_imgs:
                    continue

                tasks.extend([
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Batch Size vs Inverse Loss',
                            'Batch Size',
                            '1/loss',
                            'bss',
                            slice(None),
                            'perfs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'batch_vs_perf')
                        ),
                        dict(),
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Batch Size vs Inverse Loss Deriv.',
                            'Batch Size',
                            '1/loss Deriv wrt. BS',
                            'bss',
                            slice(None),
                            'perf_derivs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'batch_vs_perf_deriv')
                        ),
                        dict(),
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            'Batch Size vs Inverse Loss Deriv. (Smoothed)',
                            'Batch Size',
                            '1/loss Deriv wrt. LR',
                            'bss',
                            slice(None),
                            'smoothed_perf_derivs',
                            slice(trial, trial+1),
                            os.path.join(real_out, 'batch_vs_smoothed_perf_deriv')
                        ),
                        dict(),
                        1
                    ),
                ])

    # TODO other stuff

    sug_imports = ('ignite_simple.analysis',)

    dispatcher.dispatch(tasks, cores, sug_imports)

    logger.info('Finished analyzing %s', folder)
