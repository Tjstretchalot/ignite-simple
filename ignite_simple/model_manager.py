"""This module is meant to be responsible for all of the training performed
on an identical model, dataset, and loss. Specifically, it decides on the
folder structure, collates results for analysis, and handles archiving
old data.
"""
import typing
import psutil
import os
import datetime
import logging
import logging.config
import ignite_simple  # pylint: disable=unused-import
import ignite_simple.hyperparams as hparams
import ignite_simple.analarams as aparams
import ignite_simple.utils as utils
import ignite_simple.tuner as tuner
import ignite_simple.trainer as trainer
from ignite_simple.analysis import analyze
from ignite.engine import Events
from ignite.contrib.handlers import CustomPeriodicEvent
from ignite_simple.range_finder import autosmooth
import json
import numpy as np
import multiprocessing as mp
import time
import torch
import shutil

def _snap_perf(partial_epoch, ind, losses_train, losses_val, perfs_train,
               perfs_val, num_for_metric, tnr, state):
    state.evaluator.run(utils.create_partial_loader(
        state.train_set, num_for_metric))

    metrics = state.evaluator.state.metrics
    losses_train[ind[0]] = float(metrics['loss'])
    perfs_train[ind[0]] = float(metrics['perf'])

    state.evaluator.run(utils.create_partial_loader(
        state.val_set, num_for_metric))

    metrics = state.evaluator.state.metrics
    losses_val[ind[0]] = float(metrics['loss'])
    perfs_val[ind[0]] = float(metrics['perf'])


def _update_partial_epoch(partial_epoch, epochs, ind, tnr, state):
    num_in_batch = len(tnr.state.batch[0])
    epoch_size = len(state.train_set)

    try:
        epochs[ind[0]] = (float(num_in_batch) * tnr.state.iteration) / float(epoch_size)
    except IndexError:
        print(f'epoch={tnr.state.epoch}, iteration={tnr.state.iteration}')
        raise

def _increment_ind(ind, tnr, state):
    ind[0] += 1

def _reset_partial_epoch(partial_epoch, tnr, state):
    partial_epoch[0] = 0

def _save_result(filen, loss_train, loss_val, perf_train, perf_val, tnr,
                 state):
    with open(filen, 'w') as outfile:
        json.dump({
            'epochs': int(tnr.state.epoch),
            'loss_train': float(loss_train[0]),
            'loss_val': float(loss_val[0]),
            'perf_train': float(perf_train[0]),
            'perf_val': float(perf_val[0])
        }, outfile)

def _savez_compressed(filen, tnr, state, **kwargs):
    np.savez_compressed(filen, **kwargs)

def _save_model(filen, tnr, state):
    torch.save(state.unstripped_model, filen)

def _init_cpe(cpe, tnr):
    cpe.attach(tnr)

def _trial(model_loader, dataset_loader, loss_loader, trial_folder,
           with_throughtime, accuracy_style, lr_start, lr_end, batch_size,
           cycle_time_epochs, num_epochs):
    if os.path.exists('logging-worker.conf'):
        logging.config.fileConfig('logging-worker.conf')

    logger = logging.getLogger(__name__)
    logger.debug('Starting trial: accuracy_style=%s, lr_start=%s, lr_end=%s, '
                 + 'batch_size=%s, cycle_time_epochs=%s, num_epochs=%s',
                 accuracy_style, lr_start, lr_end, batch_size,
                 cycle_time_epochs, num_epochs)

    os.makedirs(trial_folder)

    train_set, val_set = utils.invoke(dataset_loader)
    len_train_set = len(train_set)
    len_val_set = len(val_set)
    del train_set
    del val_set

    final_loss_train = np.zeros(1)
    final_loss_val = np.zeros(1)
    final_perf_train = np.zeros(1)
    final_perf_val = np.zeros(1)
    final_ind = [0]
    final_partial_epoch = [1]

    handlers = [
        (
            Events.STARTED,
            (
                __name__, '_save_model',
                (
                    os.path.join(trial_folder, 'model_init.pt'),
                ),
                dict()
            )
        ),
        (
            Events.COMPLETED,
            (
                __name__, '_snap_perf',
                (
                    final_partial_epoch,
                    final_ind,
                    final_loss_train,
                    final_loss_val,
                    final_perf_train,
                    final_perf_val,
                    min(len_train_set, len_val_set)
                ),
                dict()
            ),
        ),
        (
            Events.COMPLETED,
            (
                __name__, '_save_result',
                (
                    os.path.join(trial_folder, 'result.json'),
                    final_loss_train, final_loss_val, final_perf_train,
                    final_perf_val
                ),
                dict()
            )
        ),
        (
            Events.COMPLETED,
            (
                __name__, '_save_model',
                (
                    os.path.join(trial_folder, 'model.pt'),
                ),
                dict()
            )
        ),
    ]

    initializer = None

    if with_throughtime:
        iters_per = int(len_train_set / batch_size)
        iters_total = iters_per * num_epochs

        samples_per = min(10, iters_per)
        iters_per_sample = iters_per // samples_per

        if iters_per_sample < 1:
            iters_per_sample = 1
            samples_per = iters_per

        cpe = CustomPeriodicEvent(n_iterations=iters_per_sample)
        snap_event = getattr(cpe.Events, f'ITERATIONS_{iters_per_sample}_COMPLETED')  # pylint: disable=no-member
        initializer = (
            __name__, '_init_cpe', (cpe,), dict()
        )

        samples_total = iters_total // iters_per_sample

        num_for_metric = min(1024, len_val_set, len_train_set)
        settings = np.array([num_for_metric])
        epochs = np.zeros(samples_total)
        loss_train = np.zeros(samples_total)
        loss_val = np.zeros(samples_total)
        perf_train = np.zeros(samples_total)
        perf_val = np.zeros(samples_total)
        ind = [0]
        partial_epoch = [0]
        handlers.extend([
            (
                snap_event,
                (
                    __name__, '_update_partial_epoch',
                    (
                        partial_epoch, epochs, ind
                    ),
                    dict()
                ),
            ),
            (
                snap_event,
                (
                    __name__, '_snap_perf',
                    (
                        partial_epoch,
                        ind,
                        loss_train,
                        loss_val,
                        perf_train,
                        perf_val,
                        num_for_metric,
                    ),
                    dict()
                ),
            ),
            (
                snap_event,
                (
                    __name__, '_increment_ind',
                    (
                        ind,
                    ),
                    dict()
                ),
            ),
            (
                Events.EPOCH_COMPLETED,
                (
                    __name__, '_reset_partial_epoch',
                    (
                        partial_epoch,
                    ),
                    dict()
                )
            ),
            (
                Events.COMPLETED,
                (
                    __name__, '_savez_compressed',
                    (
                        os.path.join(trial_folder, 'throughtime.npz'),
                    ),
                    {
                        'settings': settings,
                        'epochs': epochs,
                        'losses_train': loss_train,
                        'losses_val': loss_val,
                        'perfs_train': perf_train,
                        'perfs_val': perf_val
                    }
                )
            )
        ])

    tnr_settings = trainer.TrainSettings(
        accuracy_style, model_loader, loss_loader,
        (
            'ignite_simple.utils', 'task_loader',
            (dataset_loader, batch_size, True, True),
            dict()
        ),
        handlers,
        initializer,
        lr_start, lr_end, cycle_time_epochs, num_epochs
    )

    try:
        trainer.train(tnr_settings)
    except:
        traceback.print_exc()
        logger.exception('Exception encountered while training during sweep')
        raise

    logger.debug('Finished trial')

def train(model_loader: typing.Tuple[str, str, tuple, dict],
          dataset_loader: typing.Tuple[str, str, tuple, dict],
          loss_loader: typing.Tuple[str, str, tuple, dict],
          folder: str,
          hyperparameters: typing.Union[str, hparams.HyperparameterSettings],
          analysis: typing.Union[str, aparams.AnalysisSettings],
          allow_later_analysis_up_to: typing.Union[
              str, aparams.AnalysisSettings],
          accuracy_style: str,
          trials: int, is_continuation: bool,
          history_folder: str, cores: typing.Union[str, int],
          trials_strict: bool = False) -> None:
    """Trains the given model on the given dataset with the given loss by
    finding and saving or loading the hyperparameters with the given settings,
    performing the given analysis but gathering sufficient data to later
    analyze up to the specified amount.

    The folder structure is as follows:

    .. code:: none

        folder/
            hparams/
                the result from tuner.tune
            trials/
                i/  (where i=0,1,...)
                    result.json
                        note that for classification tasks perf is accuracy,
                        and in other tasks it is inverse loss. Note that perf
                        is always available and higher is better, whereas for
                        loss lower is better.

                        {'loss_train': float, 'loss_val': float,
                         'perf_train': float, 'perf_val': float}
                    model_init.pt
                        the initial random initialization of the model, saved
                        with torch.save
                    model.pt
                        the model after training, saved with torch.save
                    throughtime.npz
                        this is only stored if storing training_metric_imgs.

                        settings:
                            shape (1,), where the values are:
                                - number of points randomly selected from the
                                  corresponding dataset to calculate metrics

                        epochs:
                            the partial epoch number for the samples

                        losses_train:
                            the loss for the corresponding epoch, same shape as
                            epochs, for the training dataset

                        losses_val:
                            the loss for the corresponding epoch, same shape as
                            epochs, for the validation dataset

                        perfs_train:
                            in classification tasks this is fractional accuracy
                            in other tasks, this is inverse loss

                            the performance at the corresponding epoch, same
                            shape as epochs, for the training dataset

                        perfs_val:
                            in classification tasks this is fractional accuracy
                            in other tasks, this is inverse loss

                            the performance at the corresponding epoch, same
                            shape as epochs, for the validation dataset
            results.npz
                The trials/result.json except concatenated together for easier
                loading

                final_loss_train:
                    shape (trials,)

                final_loss_val:
                    shape (trials,)

                final_perf_train:
                    shape (trials,)

                final_perf_val:
                    shape (trials,)

            throughtimes.npz
                the trials/throughtime.npz, if available, stacked for easier
                loading

                settings:
                    shape (1,) the number of points used for gathering metrics
                    at each sample

                epochs:
                    shape (samples,) the epoch that corresponds to each sample
                    during training. this is a float, since we may sample
                    multiple times per epoch

                losses_train:
                    shape (trials, samples)

                losses_train_smoothed:
                    shape (trials, samples)

                losses_val:
                    shape (trials, samples)

                losses_val_smoothed:
                    shape (trials, samples)

                perfs_train:
                    shape (trials, samples)

                perfs_train_smoothed:
                    shape (trials, samples)

                perfs_val:
                    shape (trials, samples)

                perfs_val_smoothed:
                    shape (trials, samples)

    :param model_loader: (module, attrname, args, kwargs) defines where the
        callable which returns a torch.nn.Module can be found, and what
        arguments to pass to the callable to get the module. The callable
        should return a random initialization of the model.
    :param dataset_loader: (module, attrname, args, kwargs) defines where
        the callable which returns (train_set, val_set) can be found, and
        what arguments to pass to the callable to get the datasets
    :param loss_loader: (module, attrname, args, kwargs) defines where the
        callable which returns the torch.nn.Module that computes a scalar
        value which ought to be minimized, and what arguments to pass the
        callable for the loss.
    :param folder: the folder where the output should be stored
    :param hyperparameters: the hyperparameter settings or a name of a
        preset (one of `fastest`, `fast`, `slow`, and `slowest`)
    :param analysis: the analysis settings or a name of a preset
        (typically one of `none`, `text`, `images`, `animations`, `videos`),
        for a complete list see `ignite_simple.analarams`. It is always
        equivalent to set this value to `none` and then call
        analysis.reanalyze with the desired analysis
    :param allow_later_analysis_up_to: this is also an analysis settings or
        name of a preset, except this analysis isn't produced but instead
        we ensure that sufficient data is collected to perform this analysis
        if desired later. it must be at least as high as analysis
    :param accuracy_style: one of `classification`, `multiclass`, and
        `inv-loss` to describe how performance is measured. classification
        assumes one-hot labels for the output, multiclass assumes potentially
        multiple ones in the labels, and `inv-loss` uses 1/loss as the
        performance metric instead.
    :param trials: the number of trials which should be formed with the found
        settings
    :param is_continuation: if True then if folder already exists then it is
        assumed to have been the result of this function called with the same
        parameters except possible trials, and the result will be the sum of
        the existing trials plus the new trials to perform. If this is False
        and the folder already exists, it will be moved into history_folder
        where the name is the current timestamp.
    :param history_folder: where to store the old folders if they are found
        when is_continuation is False.
    :param cores: either an integer for the number of physical cores that are
        available for training, or the string 'all' for the number of cores
        to be auto-detected and used.
    :param trials_strict: if False, then this will use all available resources
        to compute trials such that this completes in approximately the minimum
        amount of time to produce the required number of trials. This may
        result in more than the specified number of trials being run. If True,
        exactly trial runs will be performed regardless of the amount of
        available computational resources (i.e., available cores may be unused)
    """
    model_loader = utils.fix_imports(model_loader)
    dataset_loader = utils.fix_imports(dataset_loader)
    loss_loader = utils.fix_imports(loss_loader)

    hyperparameters = hparams.get_settings(hyperparameters)
    skip_analysis = analysis == 'none'
    analysis = aparams.get_settings(analysis)
    allow_later_analysis_up_to = aparams.get_settings(
        allow_later_analysis_up_to)
    if cores == 'all':
        cores = psutil.cpu_count(logical=False)
    logger = logging.getLogger(__name__)
    logger.debug('Starting trial with model args %s, %s, dataset args %s %s',
                  model_loader[2], model_loader[3], dataset_loader[2],
                  dataset_loader[3])

    if not is_continuation and os.path.exists(folder):
        tstr = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
        fname = tstr
        os.makedirs(history_folder, exist_ok=True)

        ctr = 1
        while os.path.exists(os.path.join(history_folder, fname)):
            ctr += 1
            fname = f'{tstr}_({ctr})'

        logger.info('Archiving %s to %s...', folder,
                    os.path.join(history_folder, fname))
        os.rename(folder, os.path.join(history_folder, fname))

    continuing = is_continuation and os.path.exists(folder)

    if not continuing:
        logger.info('Tuning learning rate and batch size...')
        try:
            tuner.tune(model_loader, dataset_loader, loss_loader, 'inv-loss',
                       os.path.join(folder, 'hparams'), cores,
                       hyperparameters,
                       allow_later_analysis_up_to)
        except:
            logger.exception('An error occurred while tuning')
            raise

    with open(os.path.join(folder, 'hparams', 'final.json')) as infile:
        hparam_final = json.load(infile)

    lr_start = float(hparam_final['lr_start'])
    lr_end = float(hparam_final['lr_end'])
    batch_size = int(hparam_final['batch_size'])
    cycle_time_epochs = int(hparam_final['cycle_size_epochs'])
    epochs = int(hparam_final['epochs'])

    trial_offset = 0
    if continuing:
        while os.path.exists(
                os.path.join(folder, 'trials', str(trial_offset))):
            trial_offset += 1
        logger.info('Detected %s existing trials...', trial_offset)

    if trial_offset == 0:
        continuing = False

        if trials == 0 and trials_strict:
            logger.info('No trials -> skipping collating and analyzing')
            return

    with_throughtime = allow_later_analysis_up_to.training_metric_imgs
    if cores > 1:
        logger.info('Performing at least %s trials across %s cores...',
                    trials, cores)
        processes = []
        num_trials = 0
        last_print_trial = 0
        while num_trials < trials or (not trials_strict and num_trials < cores):
            if len(processes) >= cores:
                if last_print_trial < num_trials:
                    logger.info('Started up to trial %s...', num_trials)
                    last_print_trial = num_trials

                time.sleep(0.1)
                for i in range(len(processes) - 1, -1, -1):
                    if not processes[i].is_alive():
                        processes.pop(i)
                continue

            proc = mp.Process(
                target=_trial,
                args=(
                    model_loader, dataset_loader, loss_loader,
                    os.path.join(folder, 'trials', str(trial_offset + num_trials)),
                    with_throughtime,
                    accuracy_style, lr_start, lr_end, batch_size,
                    cycle_time_epochs, epochs
                )
            )
            proc.start()
            processes.append(proc)
            num_trials += 1

        logger.info('Waiting for %s trials to complete...', len(processes))

        for proc in processes:
            proc.join()
    else:
        logger.info('Performing %s trials (single core)...', trials)
        for num_trials in range(trials):
            _trial(model_loader, dataset_loader, loss_loader,
                   os.path.join(folder, 'trials', str(trial_offset + num_trials)),
                   with_throughtime, accuracy_style, lr_start, lr_end, batch_size,
                   cycle_time_epochs, epochs)
            logger.info('Finished trial %s/%s', num_trials + 1, trials)

        num_trials = trials

    logger.info('Collating data...')

    # This is a bit messy and a bit repetitive, but trying to break it out
    # into functions would require some very arduent function signatures
    # that are all but certainly never going to be reused

    res_file = os.path.join(folder, 'results.npz')
    tt_file = os.path.join(folder, 'throughtimes.npz')

    to_collate = dict()
    skip_collate = {'settings', 'epochs'}
    skipped_sample = dict()
    if continuing:
        logger.debug('Fetching previous information...')
        with np.load(res_file) as infile:
            for key, val in infile.items():
                if key in skip_collate:
                    skipped_sample[key] = val
                    logger.debug('Stored value for %s (skip_sample)', key)
                else:
                    to_collate[key] = [val]
                    logger.debug('Stored a value for %s', key)
        os.remove(res_file)
        if with_throughtime:
            logger.debug('Fetching previous throughtime information')
            with np.load(tt_file) as infile:
                for key, val in infile.items():
                    if key in skip_collate:
                        skipped_sample[key] = val
                        logger.debug('Stored value for %s (skip_sample, tt)', key)
                    else:
                        to_collate[key] = [val]
                        logger.debug('Stored value for %s (tt)', key)

            os.remove(tt_file)

    logger.debug('Fetching trial information (trial_offset=%s, num_trials=%s)',
                 trial_offset, num_trials)
    for trial in range(trial_offset, trial_offset + num_trials):
        trial_folder = os.path.join(folder, 'trials', str(trial))
        logger.debug('Fetching info from trial %s at %s',
                     trial, trial_folder)
        with open(os.path.join(trial_folder, 'result.json'), 'r') as infile:
            inparsed = json.load(infile)
            for key, val in inparsed.items():
                key = f'final_{key}'
                logger.debug('Considering key %s from trial %s', key, trial)
                if key in skip_collate:
                    if key not in skipped_sample:
                        skipped_sample[key] = val
                        logger.debug('Storing in skipped_sample')
                    else:
                        logger.debug('Skipping')
                else:
                    to_ap = np.stack([np.array(val)])
                    if key not in to_collate:
                        to_collate[key] = [to_ap]
                        logger.debug('Storing in to_collate (first)')
                    else:
                        to_collate[key].append(to_ap)
                        logger.debug('Storing in to_collate (now have %s)',
                                     len(to_collate[key]))
        if with_throughtime:
            logger.debug('Fetching through time info from trial %s', trial)
            with np.load(os.path.join(trial_folder, 'throughtime.npz')) as infile:
                for key, val in infile.items():
                    if key in skip_collate:
                        if key not in skipped_sample:
                            skipped_sample[key] = val
                            logger.debug('Storing in skipped_sample')
                        else:
                            logger.debug('Skipping')
                    else:
                        to_ap = np.stack([np.array(val)])
                        if key not in to_collate:
                            to_collate[key] = [to_ap]
                            logger.debug('Storing in to_collate (first)')
                        else:
                            to_collate[key].append(to_ap)
                            logger.debug('Storing in to_collate (now have %s)',
                                         len(to_collate[key]))

    logger.debug('Collating collected values')
    for key, val in tuple(to_collate.items()):
        logger.debug('Collating %s', key)
        to_collate[key] = np.concatenate(val, axis=0)

    np.savez_compressed(
        res_file,
        final_loss_train=to_collate['final_loss_train'],
        final_loss_val=to_collate['final_loss_val'],
        final_perf_train=to_collate['final_perf_train'],
        final_perf_val=to_collate['final_perf_val']
    )
    if with_throughtime:
        np.savez_compressed(
            tt_file,
            settings=skipped_sample['settings'],
            epochs=skipped_sample['epochs'],
            losses_train=to_collate['losses_train'],
            losses_train_smoothed=autosmooth(to_collate['losses_train']),
            losses_val=to_collate['losses_val'],
            losses_val_smoothed=autosmooth(to_collate['losses_val']),
            perfs_train=to_collate['perfs_train'],
            perfs_train_smoothed=autosmooth(to_collate['perfs_train']),
            perfs_val=to_collate['perfs_val'],
            perfs_val_smoothed=autosmooth(to_collate['perfs_val']),
        )

    if continuing and os.path.exists(os.path.join(folder, 'analysis')):
        logger.info('Cleaning old analysis folder...')
        shutil.rmtree(os.path.join(folder, 'analysis'))

    if not skip_analysis:
        logger.debug('Model manager delegating to analysis')
        analyze(dataset_loader, loss_loader, folder, analysis, accuracy_style, cores)

    logger.debug('Model manager exitted normally')
