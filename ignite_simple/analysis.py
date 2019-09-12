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
import ignite_simple.text_analysis as text_analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from functools import reduce
import operator
import os
import psutil
import scipy.special
import warnings
import torch
import pca3dvis.worker
import pca3dvis.pcs

REDUCTIONS = ('each', 'mean', 'mean_with_errorbars', 'mean_with_fillbtwn',
              'lse')

def _rawplot_filter(folder, images):
    """This function gives the filter to pass to futils.save_fig for the images
    to save. If folder or images is None, this always returns true (i.e.,
    this filters nothing). Otherwise, this returns true if an images filepath
    relative to the given folder is in images. This assumes that images uses
    forward slashes for separators regardless of os.path.sep

    Example:

    folder out/

    images = { john.png, doe/pong.pdf }

    filter result for out/john.png -> true
    filter result for out/john.pdf -> false
    filter result for out/pong.pdf -> false
    filter result for out/doe/pong.pdf -> true
    filter result for john.png -> false


    :param folder: the folder which the images set contains paths relative to,
        or None to just return a no-op filter
    :type folder: Optional[str]
    :param images: the set of images which should be produced or None to return
        a no-op filter
    :type images: Optional[set[str]]
    """
    if folder is None or images is None:
        return lambda x: True

    folder = os.path.abspath(folder)
    def filter_(img: str):
        img = os.path.abspath(img)

        if not img.startswith(folder):
            return False
        img = img[len(folder) + len(os.path.sep):]

        img = img.replace(os.path.sep, '/')
        return img in images

    return filter_

def _rawplot(infile, title, xlab, ylab, x_varname, x_slice, y_varname, y_slice,
             outfile_wo_ext, reduction='none', folder=None, images=None):
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

        * `lse`

            take the logsumexp over the first dimension

    :param optional[str] folder: the "current" folder or equivalent, which
        contains  the hparams and trials folders. Only required if images
        is not None
    :param optional[set[str]] images: if not None then only images whose
        filepath relative to folder will be produced.
    """
    with np.load(infile) as nin:
        xs = nin[x_varname][x_slice]
        ys = nin[y_varname][y_slice]

    _filter = _rawplot_filter(folder, images)

    new_shape = list(i for i in ys.shape if i != 1)
    ys = ys.reshape(new_shape)

    new_shape_x = list(i for i in xs.shape if i != 1)
    xs = xs.reshape(new_shape_x)
    if len(new_shape_x) != 1:
        xs = xs[0]  # take first

    if len(new_shape) > 1:
        if reduction == 'mean':
            ys = ys.mean(0)
        elif reduction == 'lse':
            old_settings = np.seterr(under='ignore')
            ys = scipy.special.logsumexp(ys, axis=0)
            np.seterr(**old_settings)
        elif reduction in ('mean_with_errorbars', 'mean_with_fillbtwn'):
            stds = ys.std(0)
            means = ys.mean(0)

            errs = 1.96 * stds
            errs_low = means - errs
            errs_high = means + errs
            ys = means
        elif reduction == 'each':
            pass
        else:
            raise ValueError(
                f'cannot reduce shape {new_shape} with {reduction}')
    else:
        reduction = 'none'

    warnings.simplefilter('ignore', UserWarning)
    fig, ax = plt.subplots()
    fig.set_figwidth(19.2)
    fig.set_figheight(10.8)
    fig.set_dpi(100)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if reduction in ('none', 'mean', 'mean_with_fillbtwn', 'lse'):
        ax.plot(xs, ys)

        if reduction == 'mean_with_fillbtwn':
            ax.fill_between(xs, errs_low, errs_high, color='grey', alpha=0.4)
    elif reduction == 'mean_with_errorbars':
        ax.errorbar(xs, ys, errs)
    elif reduction == 'each':
        for y in ys:
            ax.plot(xs, y)
    else:
        raise ValueError(f'unknown reduction {reduction}')

    futils.save_fig(fig, ax, title, outfile_wo_ext, _filter)

    plt.close(fig)

SELECTED_RANGE_YS = (
    ('perfs', '1/(loss+1)', '1/(loss+1)'),
    ('smoothed_perfs', '1/(loss+1) (Smoothed)', '1/(loss+1)'),
    ('perf_derivs', '1/(loss + 1) (Derivative)', '1/(loss+1) deriv.'),
    ('smoothed_perf_derivs', '1/(loss + 1) (Deriv. Sm.)', '1/(loss+1) deriv.')
)
def _highlight_selected_range(xs, ys, rge, x_lab_short, x_lab_long,
                              y_lab_short, y_lab_long, outfile_wo_ext,
                              folder, images):
    stds = ys.std(0)
    means = ys.mean(0)

    errs = 1.96 * stds
    errs_low = means - errs
    errs_high = means + errs

    warnings.simplefilter('ignore', UserWarning)
    fig, ax = plt.subplots()
    fig.set_figwidth(19.2)
    fig.set_figheight(10.8)
    fig.set_dpi(100)

    ax.set_xlabel(x_lab_long)
    ax.set_ylabel(y_lab_long)

    ax.plot(xs, means)
    ax.fill_between(xs, errs_low, errs_high, color='grey', alpha=0.4)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if xlim[0] < rge[0]:
        ax.fill_between((xlim[0], rge[0]), (ylim[0], ylim[0]),
                        (ylim[1], ylim[1]), color='black', alpha=0.4)

    if xlim[1] > rge[1]:
        ax.fill_between((rge[1], xlim[1]), (ylim[0], ylim[0]),
                        (ylim[1], ylim[1]), color='black', alpha=0.4)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    futils.save_fig(fig, ax, futils.make_vs_title(x_lab_short, y_lab_short),
                    outfile_wo_ext, _rawplot_filter(folder, images))

    plt.close(fig)

def _highlight_selected_lr_range(lr_vs_perf: str, y_lab_long: str,
                                 y_lab_short: str, y_varname: str,
                                 outfile_wo_ext: str,
                                 folder: str = None,
                                 images: typing.Set[str] = None):
    with np.load(lr_vs_perf) as infile:
        xs = infile['lrs']
        ys = infile[y_varname]
        rge = infile['lr_range']
    _highlight_selected_range(xs, ys, rge, 'LR', 'Learning Rate', y_lab_short,
                              y_lab_long, outfile_wo_ext, folder, images)

def _highlight_selected_batch_range(batch_vs_perf: str, y_lab_long: str,
                                    y_lab_short: str, y_varname: str,
                                    outfile_wo_ext: str,
                                    folder: str = None,
                                    images: typing.Set[str] = None):
    with np.load(batch_vs_perf) as infile:
        xs = infile['bss']
        ys = infile[y_varname]
        rge = infile['bs_range']
    _highlight_selected_range(xs, ys, rge, 'BS', 'Batch Size', y_lab_short,
                              y_lab_long, outfile_wo_ext, folder, images)

def _pca3dvis_model(dataset_loader, model_file, outfolder, use_train,
                    accuracy_style, draft):
    train_set, val_set = utils.invoke(dataset_loader)
    dset = train_set if use_train else val_set
    model = torch.load(model_file)

    num_pts = min(len(dset), 1024)
    pts, raw_lbls = next(iter(utils.create_partial_loader(
        dset, num_pts, num_pts)))

    if accuracy_style == 'classification':
        lbls = raw_lbls.numpy()
        markers = [
            (
                np.ones(num_pts, dtype='bool'),
                {
                    'c': lbls,
                    'cmap': plt.get_cmap('Set1'),
                    's': 20 if draft else 30,
                    'marker': 'o',
                    'norm': mcolors.Normalize(lbls.min(), lbls.max())
                }
            )
        ]
    else:
        # possibilities: color by loss, color by correctness, color by norms
        lbls = np.ones(num_pts, 'int32')
        markers = [
            (
                np.ones(num_pts, 'bool'),
                {'s': 20 if draft else 30, 'c': 'tab:red'}
            )
        ]
    with torch.set_grad_enabled(False):
        _, hidacts = model(pts)


    hidacts = [ha.detach().numpy() for ha in hidacts]
    for i in range(len(hidacts)):  # pylint: disable=consider-using-enumerate
        hidacts[i] = hidacts[i].reshape((num_pts, reduce(operator.mul, hidacts[i].shape[1:])))
        if hidacts[i].shape[1] < 3:
            new_ha = np.zeros((num_pts, 3), dtype=hidacts[i].dtype)
            new_ha[:, :hidacts[i].shape[1]] = hidacts[i]
            hidacts[i] = new_ha

    titles = [f'Hidden Layer {i}' for i, _ in enumerate(hidacts)]
    titles[0] = 'Input'
    if len(titles) == 3:
        titles[1] = 'Hidden Layer'
    titles[-1] = 'Output'

    traj = pca3dvis.pcs.get_pc_trajectory(hidacts, lbls)

    logger = logging.getLogger(__name__)
    logger.info('Generating pca3dvis for hidden activations (draft=%s, train=%s, out=%s)...',
                draft, use_train, outfolder)
    pca3dvis.worker.generate(traj, markers, titles, outfolder, draft)
    logger.info('Finished generating video at %s', outfolder)

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
                            lr_vs_smoothed_perf.(img)
                            lr_vs_perf_deriv.(img)
                            lr_vs_smoothed_perf_deriv.(img)

                        lr_vs_perf_*.(img)
                        lr_vs_smoothed_perf_*.(img)
                        lr_vs_perf_deriv_*.(img)
                        lr_vs_smoothed_perf_deriv_*.(img)
                        lr_vs_lse_smoothed_perf_then_deriv_*.(img)
                        lr_vs_lse_smoothed_perf_then_deriv_then_smooth_*.(img)
                    batch/
                        i/ (where i=0,1,...)
                            batch_vs_perf.(img)
                            batch_vs_smoothed_perf.(img)
                            batch_vs_perf_deriv.(img)
                            batch_vs_smoothed_perf_deriv.(img)

                        batch_vs_perf_*.(img)
                        batch_vs_smoothed_perf_*.(img)
                        batch_vs_perf_derivs_*.(img)
                        batch_vs_smoothed_perf_derivs_*.(img)
                        batch_vs_lse_smoothed_perf_then_derivs_*.(img)

                    TODO videos & animations
                trials/
                    i/  (where i=0,1,...)
                        epoch_vs_loss_train.(img) (*)
                        epoch_vs_loss_val.(img) (*)
                        epoch_vs_perf_train.(img) (*)
                        epoch_vs_perf_val.(img) (*)

                        pca3dvis_train_draft/
                            Only produced if settings.typical_run_pca3dvis and
                            settings.typical_run_pca3dvis_draft are set, and
                            only done on trial 0
                        pca3dvis_train/
                            Only produced if settings.typical_run_pca3dvis and
                            not settings.typical_run_pca3dvis_draft, and only
                            done on trial 0

                    epoch_vs_loss_train_*.(img) (*)
                    epoch_vs_loss_val_*.(img) (*)
                    epoch_vs_smoothed_loss_train_*.(img) (*)
                    epoch_vs_smoothed_loss_val_*.(img) (*)
                    epoch_vs_perf_train_*.(img) (*)
                    epoch_vs_smoothed_perf_train_*.(img) (*)
                    epoch_vs_perf_val_*.(img) (*)
                    epoch_vs_smoothed_perf_val_*.(img) (*)


                    (*)
                        Only produced if throughtime.npz is available for
                        the trial and settings.training_metric_images is
                        set

                    TODO more summary of trials
                    TODO text & videos & animations

                html/
                    See text_analysis.py for details

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
    if settings.suppress_extra_images:
        imgs_to_produce = text_analysis.HTML_REFD_IMAGES
    else:
        imgs_to_produce = None

    filter_folder = os.path.join(folder, 'analysis')
    imgs_filter = _rawplot_filter(filter_folder, imgs_to_produce)

    logger = logging.getLogger(__name__)

    logger.info('Analyzing %s...', folder)

    perf_name = (
        'Accuracy' if accuracy_style in ('classification, multiclass')
        else 'Inverse Loss'
    )
    perf_name_short = {
        'classification': 'Accuracy (%)',
        'multiclass': 'Subset Accuracy Score',
    }.get(accuracy_style, '1/(loss+1)')

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

                have_imgs = futils.fig_exists(
                    os.path.join(real_out, 'lr_vs_perf'), imgs_filter)
                if have_imgs:
                    continue

                tasks.extend([
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title('Learning Rate', 'Inverse Loss'),
                            'Learning Rate',
                            '1/(loss+1)',
                            'lrs',
                            slice(None),
                            'perfs',
                            slice(trial, trial + 1),
                            os.path.join(real_out, 'lr_vs_perf'),
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title('Learning Rate', 'Inverse Loss'),
                            'Learning Rate',
                            '1/(loss+1)',
                            'lrs',
                            slice(None),
                            'smoothed_perfs',
                            slice(trial, trial + 1),
                            os.path.join(real_out, 'lr_vs_smoothed_perf')
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate', 'Inverse Loss Deriv.'),
                            'Learning Rate',
                            '1/(loss+1) Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'perf_derivs',
                            slice(trial, trial + 1),
                            os.path.join(real_out, 'lr_vs_perf_deriv')
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate',
                                'Inverse Loss Deriv. (Smoothed)'),
                            'Learning Rate',
                            '1/(loss+1) Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'smoothed_perf_derivs',
                            slice(trial, trial + 1),
                            os.path.join(real_out, 'lr_vs_smoothed_perf_deriv')
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                ])

            for reduction in REDUCTIONS:
                have_imgs = futils.fig_exists(
                    os.path.join(out_folder, f'lr_vs_perf_{reduction}'), imgs_filter)
                if have_imgs:
                    continue

                tasks.extend([
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate', 'Inverse Loss'),
                            'Learning Rate',
                            '1/(loss+1)',
                            'lrs',
                            slice(None),
                            'perfs',
                            slice(None),
                            os.path.join(out_folder, 'lr_vs_perf_' + reduction),
                            reduction
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate', 'Inverse Loss (Smoothed)'),
                            'Learning Rate',
                            '1/(loss+1)',
                            'lrs',
                            slice(None),
                            'smoothed_perfs',
                            slice(None),
                            os.path.join(out_folder, 'lr_vs_smoothed_perf_' + reduction),
                            reduction
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate', 'Inverse Loss Deriv.'),
                            'Learning Rate',
                            '1/(loss+1) Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'perf_derivs',
                            slice(None),
                            os.path.join(out_folder, 'lr_vs_perf_deriv_' + reduction),
                            reduction
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                    dispatcher.Task(
                        __name__,
                        '_rawplot',
                        (
                            source,
                            futils.make_vs_title(
                                'Learning Rate',
                                'Inverse Loss Deriv. (Smoothed)'),
                            'Learning Rate',
                            '1/(loss+1) Deriv wrt. LR',
                            'lrs',
                            slice(None),
                            'smoothed_perf_derivs',
                            slice(None),
                            os.path.join(
                                out_folder,
                                'lr_vs_smoothed_perf_deriv_' + reduction),
                            reduction
                        ),
                        {
                            'folder': filter_folder,
                            'images': imgs_to_produce
                        },
                        1
                    ),
                ])

            if not futils.fig_exists(
                    os.path.join(out_folder, 'lr_vs_lse_smoothed_perf_then_deriv'),
                    imgs_filter):
                tasks.append(dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'LR', 'Deriv of LSE of Smoothed 1/(loss+1)'),
                        'Learning Rate',
                        '1/(loss+1) Deriv wrt. LR',
                        'lrs',
                        slice(None),
                        'lse_smoothed_perf_then_derivs',
                        slice(None),
                        os.path.join(out_folder, 'lr_vs_lse_smoothed_perf_then_deriv')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ))

            if not futils.fig_exists(
                    os.path.join(out_folder, 'lr_vs_lse_smoothed_perf_then_deriv_then_smooth'),
                    imgs_filter):
                tasks.append(dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'LR', 'Deriv of LSE of Smoothed 1/(loss+1) (Smoothed)'),
                        'Learning Rate',
                        '1/(loss+1) Deriv wrt. LR',
                        'lrs',
                        slice(None),
                        'lse_smoothed_perf_then_derivs_then_smooth',
                        slice(None),
                        os.path.join(out_folder, 'lr_vs_lse_smoothed_perf_then_deriv_then_smooth')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ))

            for y_varname, y_lab_long, y_lab_short in SELECTED_RANGE_YS:
                outfile_wo_ext = os.path.join(
                    out_folder, f'lr_range_{y_varname}')
                if futils.fig_exists(outfile_wo_ext, imgs_filter):
                    continue
                tasks.append(dispatcher.Task(
                    __name__,
                    '_highlight_selected_lr_range',
                    (
                        source,
                        y_lab_long,
                        y_lab_short,
                        y_varname,
                        outfile_wo_ext
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ))

        source = os.path.join(folder, 'hparams', 'bs_vs_perf.npz')
        out_folder = batch_folder
        with np.load(source) as infile:
            num_trials = infile['perfs'].shape[0]

        for trial in range(num_trials):
            real_out = os.path.join(out_folder, str(trial))
            os.makedirs(real_out, exist_ok=True)

            have_imgs = futils.fig_exists(
                os.path.join(real_out, 'batch_vs_perf'), imgs_filter)
            if have_imgs:
                continue

            tasks.extend([
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title('Batch Size', 'Inverse Loss'),
                        'Batch Size',
                        '1/(loss+1)',
                        'bss',
                        slice(None),
                        'perfs',
                        slice(trial, trial + 1),
                        os.path.join(real_out, 'batch_vs_perf')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss (Smoothed)'),
                        'Batch Size',
                        '1/(loss+1)',
                        'bss',
                        slice(None),
                        'smoothed_perfs',
                        slice(trial, trial + 1),
                        os.path.join(real_out, 'batch_vs_smoothed_perf')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss Deriv.'),
                        'Batch Size',
                        '1/(loss+1) Deriv wrt. BS',
                        'bss',
                        slice(None),
                        'perf_derivs',
                        slice(trial, trial + 1),
                        os.path.join(real_out, 'batch_vs_perf_deriv')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss Deriv. (Smoothed)'),
                        'Batch Size',
                        '1/(loss+1) Deriv wrt. LR',
                        'bss',
                        slice(None),
                        'smoothed_perf_derivs',
                        slice(trial, trial + 1),
                        os.path.join(real_out, 'batch_vs_smoothed_perf_deriv')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
            ])

        for reduction in REDUCTIONS:
            have_imgs = futils.fig_exists(
                os.path.join(out_folder, f'batch_vs_perf_{reduction}'))
            if have_imgs:
                continue
            tasks.extend([
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title('Batch Size', 'Inverse Loss'),
                        'Batch Size',
                        '1/(loss+1)',
                        'bss',
                        slice(None),
                        'perfs',
                        slice(None),
                        os.path.join(out_folder, 'batch_vs_perf_' + reduction),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss (Smoothed)'),
                        'Batch Size',
                        '1/(loss+1)',
                        'bss',
                        slice(None),
                        'smoothed_perfs',
                        slice(None),
                        os.path.join(out_folder, 'batch_vs_smoothed_perf_' + reduction),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss Deriv.'),
                        'Batch Size',
                        '1/(loss+1) Deriv wrt. BS',
                        'bss',
                        slice(None),
                        'perf_derivs',
                        slice(None),
                        os.path.join(out_folder, 'batch_vs_perf_deriv_' + reduction),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        source,
                        futils.make_vs_title(
                            'Batch Size', 'Inverse Loss Deriv. (Smoothed)'),
                        'Batch Size',
                        '1/(loss+1) Deriv wrt. BS',
                        'bss',
                        slice(None),
                        'smoothed_perf_derivs',
                        slice(None),
                        os.path.join(out_folder, 'batch_vs_smoothed_perf_deriv_' + reduction),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
            ])

        if not futils.fig_exists(
                os.path.join(out_folder, 'batch_vs_lse_smoothed_perf_then_deriv'),
                imgs_filter):
            tasks.append(dispatcher.Task(
                __name__,
                '_rawplot',
                (
                    source,
                    futils.make_vs_title(
                        'BS', 'Deriv of LSE of Smoothed 1/(loss+1)'),
                    'Batch Size',
                    '1/(loss+1) Deriv wrt. BS',
                    'bss',
                    slice(None),
                    'lse_smoothed_perf_then_derivs',
                    slice(None),
                    os.path.join(out_folder, 'batch_vs_lse_smoothed_perf_then_deriv')
                ),
                {
                    'folder': filter_folder,
                    'images': imgs_to_produce
                },
                1
            ))


        for y_varname, y_lab_long, y_lab_short in SELECTED_RANGE_YS:
            outfile_wo_ext = os.path.join(
                out_folder, f'batch_range_{y_varname}')
            if futils.fig_exists(outfile_wo_ext, imgs_filter):
                continue
            tasks.append(dispatcher.Task(
                __name__,
                '_highlight_selected_batch_range',
                (
                    source,
                    y_lab_long,
                    y_lab_short,
                    y_varname,
                    outfile_wo_ext
                ),
                {
                    'folder': filter_folder,
                    'images': imgs_to_produce
                },
                1
            ))
    if settings.training_metric_imgs:
        trials = -1
        trials_source_folder = os.path.join(folder, 'trials')
        while os.path.exists(os.path.join(trials_source_folder, str(trials + 1))):
            trials += 1
        trial_out_folder = os.path.join(folder, 'analysis', 'trials')

        for trial in range(trials):
            trial_src = os.path.join(trials_source_folder, str(trial), 'throughtime.npz')
            if not os.path.exists(trial_src):
                continue

            trial_out = os.path.join(trial_out_folder, str(trial))
            have_imgs = futils.fig_exists(
                os.path.join(trial_out, f'epoch_vs_loss_train'), imgs_filter)
            if have_imgs:
                continue

            os.makedirs(trial_out, exist_ok=True)
            tasks.extend([
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Train)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_train',
                        slice(None),
                        os.path.join(trial_out, 'epoch_vs_loss_train')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Validation)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_val',
                        slice(None),
                        os.path.join(trial_out, 'epoch_vs_loss_val')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Train)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_train',
                        slice(None),
                        os.path.join(trial_out, 'epoch_vs_perf_train')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Validation)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_val',
                        slice(None),
                        os.path.join(trial_out, 'epoch_vs_perf_val')
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
            ])

        trial_out = trial_out_folder
        trial_src = os.path.join(folder, 'throughtimes.npz')
        for reduction in REDUCTIONS:
            have_imgs = futils.fig_exists(
                os.path.join(trial_out, f'epoch_vs_loss_train_{reduction}'), imgs_filter)
            if have_imgs:
                continue

            os.makedirs(trial_out, exist_ok=True)

            tasks.extend([
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Train)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_train',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_loss_train_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Train, Smoothed)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_train_smoothed',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_smoothed_loss_train_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Validation)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_val',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_loss_val_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', 'Loss (Val, Smoothed)'),
                        'Epoch',
                        'Loss',
                        'epochs',
                        slice(None),
                        'losses_val_smoothed',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_smoothed_loss_val_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Train)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_train',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_perf_train_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Smoothed, Train)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_train_smoothed',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_smoothed_perf_train_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Validation)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_val',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_perf_val_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
                dispatcher.Task(
                    __name__,
                    '_rawplot',
                    (
                        trial_src,
                        futils.make_vs_title('Epoch', f'{perf_name} (Val, Smoothed)'),
                        'Epoch',
                        perf_name_short,
                        'epochs',
                        slice(None),
                        'perfs_val_smoothed',
                        slice(None),
                        os.path.join(trial_out, f'epoch_vs_smoothed_perf_val_{reduction}'),
                        reduction
                    ),
                    {
                        'folder': filter_folder,
                        'images': imgs_to_produce
                    },
                    1
                ),
            ])

    if (settings.typical_run_pca3dvis
            and os.path.exists(os.path.join(folder, 'trials', '0'))):
        model_file = os.path.join(folder, 'trials', '0', 'model.pt')
        model = torch.load(model_file)
        train_set, _ = utils.invoke(dataset_loader)

        example_item = train_set[0][0]
        example_item = torch.unsqueeze(example_item, 0)

        example_out = model(example_item)
        if isinstance(example_out, tuple):
            # we have an unstripped model!
            outfolder = (
                os.path.join(
                    folder, 'analysis', 'trials', '0', 'pca3dvis_train_draft')
                if settings.typical_run_pca3dvis_draft else
                os.path.join(
                    folder, 'analysis', 'trials', '0', 'pca3dvis_train')
            )

            if not os.path.exists(outfolder):
                tasks.append(
                    dispatcher.Task(
                        __name__,
                        '_pca3dvis_model',
                        (
                            dataset_loader,
                            model_file,
                            outfolder,
                            True,
                            accuracy_style,
                            settings.typical_run_pca3dvis_draft
                        ),
                        dict(),
                        None
                    )
                )

    tasks.append(
        dispatcher.Task(
            'ignite_simple.text_analysis',
            'text_analyze',
            (
                settings,
                folder
            ),
            dict(),
            1
        )
    )

    # TODO other stuff

    sug_imports = ('ignite_simple.analysis',)

    dispatcher.dispatch(tasks, cores, sug_imports)

    logger.info('Finished analyzing %s', folder)
