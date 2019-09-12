"""This module is responsible for tuning the learning rate and batch size for
training a module."""
import ignite_simple  # pylint: disable=unused-import
import typing
import importlib
from ignite_simple.hyperparams import HyperparameterSettings
from ignite_simple.analarams import AnalysisSettings
from ignite_simple.vary_bs_loader import BatchSizeVaryingDataLoader
from ignite_simple.range_finder import smooth_window_size, find_with_derivs
import ignite_simple.utils as utils
import ignite_simple.trainer
import torch
import torch.utils.data as data
import numpy as np
from ignite.engine import Events
import os
import uuid
import multiprocessing as mp
import logging
import logging.config
import time
import scipy.signal
import scipy.special
import json
import math
import traceback

_valldr = utils.create_partial_loader
_task_loader = utils.task_loader

NUM_TO_VAL_MAX = 64 * 3

def _store_lr_and_perf(lrs, perfs, cur_iter, num_to_val, tnr,
                       state: ignite_simple.trainer.TrainState):
    valldr = _valldr(state.train_set, num_to_val)
    state.evaluator.run(valldr)

    loss = state.evaluator.state.metrics['loss']
    if math.isnan(loss):
        # our network has died!
        lrs[cur_iter[0]:] = float('nan')
        perfs[cur_iter[0]:] = float('nan')
        tnr.terminate()
        return

    lrs[cur_iter[0]] = state.lr_scheduler.get_param()
    perfs[cur_iter[0]] = state.evaluator.state.metrics['perf']

def _increment(cur, tnr, state):
    cur[0] += 1


def _lr_vs_perf(model_loader, dataset_loader, loss_loader, outfile,
                accuracy_style, lr_start, lr_end, batch_size,
                cycle_time_epochs, warmup_lr, warmup_batch, warmup_pts):
    if os.path.exists('logging-worker.conf'):
        logging.config.fileConfig('logging-worker.conf')

    logger = logging.getLogger(__name__)
    logger.debug('LR vs Perf worker started')
    train_set, _ = utils.invoke(dataset_loader)
    task_loader = (
        __name__,
        '_task_loader',
        (dataset_loader, batch_size, True, True),
        dict())

    num_train_iters = (len(train_set) // batch_size) * (cycle_time_epochs // 2)

    warmed_up_model_loader = (
        'ignite_simple.utils',
        'model_loader_with_warmup',
        (
            model_loader,
            task_loader,
            loss_loader,
            warmup_lr,
            warmup_batch,
            warmup_pts
        ),
        dict()
    ) if warmup_pts > 0 else model_loader

    cur_iter = [0]
    num_to_val = min(NUM_TO_VAL_MAX, len(train_set))

    lrs = np.zeros(num_train_iters)
    perfs = np.zeros(num_train_iters)

    tnr_settings = ignite_simple.trainer.TrainSettings(
        accuracy_style, warmed_up_model_loader, loss_loader,
        task_loader,
        (
            (Events.ITERATION_COMPLETED,
             (__name__, '_store_lr_and_perf',
              (lrs, perfs, cur_iter, num_to_val), dict())),
            (Events.ITERATION_COMPLETED,
             (__name__, '_increment', (cur_iter,), dict()))
        ),
        None,
        lr_start,
        lr_end,
        cycle_time_epochs,
        cycle_time_epochs // 2
    )

    try:
        ignite_simple.trainer.train(tnr_settings)
    except:
        traceback.print_exc()
        logger.exception('LR vs Perf worker encountered an exception')
        raise

    np.savez_compressed(outfile, lrs=lrs, perfs=perfs)
    logger.debug('LR vs Perf worker finished normally')

def _task_loader_bs(dataset_loader, batch_start, batch_end, epochs):
    train_set, val_set = utils.invoke(dataset_loader)
    train_loader = BatchSizeVaryingDataLoader(
        train_set, batch_start, batch_end, epochs)
    return train_set, val_set, train_loader


def _store_bs_and_perf(bss, perfs, cur, num_to_val, tnr,
                       state: ignite_simple.trainer.TrainState):
    valldr = _valldr(state.train_set, num_to_val)
    state.evaluator.run(valldr)
    perf = state.evaluator.state.metrics['perf']

    cur_bs, cur_sum, cur_num = cur[0]
    cur_ind = cur[1]

    if bss[cur_ind] != cur_bs:
        raise Exception(f'bss[{cur_ind}] = {bss[cur_ind]}, cur_bs={cur_bs}')

    bs_this = int(state.train_loader.last_iter.last_batch_size)

    if bs_this == cur_bs:
        cur_sum += perf
        cur_num += 1
        cur[0] = (cur_bs, cur_sum, cur_num)
    else:
        avg = cur_sum / cur_num
        perfs[cur_ind] = avg
        cur[0] = (bs_this, perf, 1)
        cur[1] = cur_ind + 1

        if cur[1] >= bss.shape[0]:
            raise Exception(f'after batch size {cur_bs} got batch size {bs_this} - out of range')

def _store_last_bs(perfs, cur, tnr, state):
    perfs[cur[1]] = cur[0][1] / cur[0][2]

def _batch_vs_perf(model_loader, dataset_loader, loss_loader, outfile,
                   accuracy_style, batch_start, batch_end, lr_start, lr_end,
                   cycle_time_epochs):
    if os.path.exists('logging-worker.conf'):
        logging.config.fileConfig('logging-worker.conf')

    logger = logging.getLogger(__name__)
    logger.debug('Batch vs Perf worker started')

    train_set, _ = utils.invoke(dataset_loader)

    # N = 0.5 * k * (k + 1)
    # => k^2 + k - 2N = 0
    # -> k = (-1 + sqrt(1 + 8N)) / 2
    epochs = cycle_time_epochs // 2
    train_loader = BatchSizeVaryingDataLoader(
        train_set, batch_start, batch_end, epochs)

    unique_batch_sizes = []
    miter = train_loader.dry_iter()
    for _ in miter:
        bs = int(miter.last_batch_size)
        if not unique_batch_sizes or bs != unique_batch_sizes[-1]:
            unique_batch_sizes.append(bs)

    bss = np.array(unique_batch_sizes)
    perfs = np.zeros(bss.shape, dtype='float32')

    cur = [(bss[0], 0, 0), 0]
    num_to_val = min(NUM_TO_VAL_MAX, len(train_set))

    tnr_settings = ignite_simple.trainer.TrainSettings(
        accuracy_style,
        model_loader,
        loss_loader,
        (__name__, '_task_loader_bs',
         (dataset_loader, batch_start, batch_end, epochs), dict()),
        (
            (Events.ITERATION_COMPLETED,
             (__name__, '_store_bs_and_perf',
              (bss, perfs, cur, num_to_val), dict())),
            (Events.COMPLETED,
             (__name__, '_store_last_bs', (perfs, cur), dict()))
        ),
        None,
        lr_start,
        lr_end,
        2,
        1
    )

    try:
        ignite_simple.trainer.train(tnr_settings)
    except:
        traceback.print_exc()
        logger.exception('Error encountered during BS sweep')
        raise

    np.savez_compressed(outfile, bss=bss, perfs=perfs)
    logger.debug('BS vs Perf worker finished normally')

def _store_perf(perfs, cur, num_to_val, tnr, state):
    valldr = _valldr(state.train_set, num_to_val)
    state.evaluator.run(valldr)
    perfs[cur[0]] = state.evaluator.state.metrics['perf']

def _train_with_perf(model_loader, dataset_loader, loss_loader, outfile,
                     accuracy_style, batch_size, lr_start, lr_end,
                     cycle_time_epochs, epochs, with_raw):
    train_set, _ = utils.invoke(dataset_loader)

    final_perf = np.zeros(1)
    final_ind = [0]
    handlers = [
        (Events.COMPLETED,
         (__name__, '_store_perf',
          (final_perf, final_ind, len(train_set)), dict()))
    ]
    if with_raw:
        num_iters = (len(train_set) // batch_size) * epochs
        num_to_val = min(NUM_TO_VAL_MAX, len(train_set))
        perf = np.zeros(num_iters)
        ind = [0]
        handlers.extend([
            (Events.ITERATION_COMPLETED,
             (__name__, '_store_perf',
              (perf, ind, num_to_val), dict())),
            (Events.ITERATION_COMPLETED,
             (__name__, '_increment', (ind,), dict()))
        ])

    tnr_settings = ignite_simple.trainer.TrainSettings(
        accuracy_style, model_loader, loss_loader,
        (__name__, '_task_loader',
         (dataset_loader, batch_size, True, True), dict()),
        handlers, None, lr_start, lr_end, cycle_time_epochs, epochs
    )
    ignite_simple.trainer.train(tnr_settings)

    to_save = {'final_perf': final_perf}
    if with_raw:
        to_save['perf'] = perf
    np.savez_compressed(outfile, **to_save)

def _run_and_collate(fn, kwargs, cores,
                     min_iters) -> typing.Dict[str, np.ndarray]:
    folder = str(uuid.uuid4())
    os.makedirs(folder)

    i = 0
    while i < min_iters:
        procs = []
        for procid in range(i, i + cores - 1):
            proc_kwargs = kwargs.copy()
            proc_kwargs['outfile'] = os.path.join(folder, f'{procid}.npz')
            proc = mp.Process(target=fn, kwargs=proc_kwargs)
            proc.start()
            procs.append(proc)
        i += cores - 1

        my_kwargs = kwargs.copy()
        my_kwargs['outfile'] = os.path.join(folder, f'{i}.npz')
        fn(**my_kwargs)
        i += 1

        for proc in procs:
            proc.join()

    all_lists = dict()
    with np.load(os.path.join(folder, '0.npz')) as infile:
        for key, val in infile.items():
            all_lists[key] = [val]

    os.remove(os.path.join(folder, '0.npz'))

    for j in range(1, i):
        fname = os.path.join(folder, f'{j}.npz')
        with np.load(fname) as infile:
            for key, val in infile.items():
                all_lists[key].append(val)
        os.remove(fname)

    os.rmdir(folder)
    return dict((key, np.stack(val)) for key, val in all_lists.items())

def _select_lr_from(model_loader, dataset_loader, loss_loader,
                    accuracy_style, outfile, cores, settings,
                    store_up_to, logger, cycle_time_epochs,
                    batch_size, lr_start, lr_end,
                    grads_width_only,
                    warmup_lr, warmup_batch, warmup_pts) -> typing.Tuple[int, int]:
    result = _run_and_collate(
        _lr_vs_perf, {
            'model_loader': model_loader,
            'dataset_loader': dataset_loader,
            'loss_loader': loss_loader,
            'accuracy_style': accuracy_style,
            'lr_start': lr_start,
            'lr_end': lr_end,
            'batch_size': batch_size,
            'cycle_time_epochs': cycle_time_epochs,
            'warmup_lr': warmup_lr,
            'warmup_batch': warmup_batch,
            'warmup_pts': warmup_pts
        }, cores, settings.lr_min_inits
    )

    logger.debug('Organizing and interpreting learning rate sweep...')

    lrs = result['lrs']
    lr_perfs = result['perfs']
    if np.isnan(lrs.sum()):
        clip_at = np.isnan(lrs.sum(0)).argmax()
        if clip_at > 0:
            new_lr_end = lrs[0, clip_at - 1]
        else:
            new_lr_end = (lr_start + 0.01 * (lr_end - lr_start))

        clip_at //= 2
        new_lr_end = max(lr_start + 0.05 * (lr_end - lr_start),
                         lr_start + 0.5 * (new_lr_end - lr_start))

        if new_lr_end < lr_start + 0.1 * (lr_end - lr_start):
            logger.debug(
                'Got too many nans, resweeping with lr range reduced to '
                + f'{lr_start}/{new_lr_end}')
            return _select_lr_from(
                model_loader, dataset_loader, loss_loader, accuracy_style,
                outfile, cores, settings, store_up_to, logger,
                cycle_time_epochs, batch_size, lr_start, new_lr_end,
                grads_width_only,
                warmup_lr, warmup_batch, warmup_pts)

        lrs = lrs[:, :clip_at]
        lr_perfs = lr_perfs[:, :clip_at]

    lrs = lrs[0]
    num_trials = lr_perfs.shape[0]
    window_size = smooth_window_size(lrs.shape[0])

    lr_smoothed_perfs = scipy.signal.savgol_filter(
        lr_perfs, window_size, 1)

    old_settings = np.seterr(under='ignore')
    lse_smoothed_lr_perfs = scipy.special.logsumexp(
        lr_smoothed_perfs, axis=0
    )
    np.seterr(**old_settings)
    lse_smoothed_lr_perf_then_derivs = np.gradient(lse_smoothed_lr_perfs)
    lr_perf_derivs = np.gradient(lr_perfs, axis=-1)
    smoothed_lr_perf_derivs = scipy.signal.savgol_filter(
        lr_perfs, window_size, 1, deriv=1)
    mean_smoothed_lr_perf_derivs = smoothed_lr_perf_derivs.mean(0)

    lse_smoothed_lr_perf_then_derivs_then_smooth = scipy.signal.savgol_filter(
        lse_smoothed_lr_perf_then_derivs, window_size, 1)
    lr_min, lr_max = find_with_derivs(lrs, lse_smoothed_lr_perf_then_derivs_then_smooth,
                                      grads_width_only)

    np.savez_compressed(
        outfile,
        lrs=lrs, perfs=lr_perfs,
        smoothed_perfs=lr_smoothed_perfs,
        lse_smoothed_perfs=lse_smoothed_lr_perfs,
        perf_derivs=lr_perf_derivs,
        smoothed_perf_derivs=smoothed_lr_perf_derivs,
        mean_smoothed_perf_derivs=mean_smoothed_lr_perf_derivs,
        lse_smoothed_perf_then_derivs=lse_smoothed_lr_perf_then_derivs,
        lse_smoothed_perf_then_derivs_then_smooth=lse_smoothed_lr_perf_then_derivs_then_smooth,
        lr_range=np.array([lr_min, lr_max]))

    logger.info('Learning rate range: [%s, %s) (found from %s trials)',
                lr_min, lr_max, num_trials)
    return lr_min, lr_max, window_size, num_trials


def _select_batch_size_from(model_loader, dataset_loader, loss_loader,
                            accuracy_style, mainfolder, cores, settings,
                            store_up_to, logger, cycle_time_epochs, bss,
                            collated_smoothed_bs_perf_derivs,
                            bs_min, bs_max, lr_min_over_batch,
                            lr_max_over_batch) -> int:
    settings: HyperparameterSettings
    store_up_to: AnalysisSettings

    bs_min_ind = int((bss == bs_min).argmax())
    bs_max_ind = int((bss == bs_max).argmax())

    incl_raw = store_up_to.hparam_selection_specific_imgs

    if bs_min_ind == bs_max_ind:
        logger.info('Only found a single good batch size, using that without '
                    + 'further investigation')
        return bs_min

    if bs_max_ind - bs_min_ind <= settings.batch_pts:
        logger.debug('Found %s good batch sizes and willing to try up to %s, '
                     + 'so testing all of them.', bs_max_ind - bs_min_ind,
                     settings.batch_pts)
        test_pts = bss[bs_min_ind:bs_max_ind]
    else:
        probs = collated_smoothed_bs_perf_derivs[bs_min_ind:bs_max_ind]
        old_settings = np.seterr(under='ignore')
        probs = scipy.special.softmax(probs)

        iters = 0
        while (probs < 1e-6).sum() != 0:
            if iters > 10:
                probs[:] = 1 / probs.shape[0]
                break
            probs[probs < 1e-6] = 1e-6
            probs = scipy.special.softmax(probs)
            iters += 1

        np.seterr(**old_settings)

        test_pts = np.random.choice(
            np.arange(bs_min_ind, bs_max_ind), settings.batch_pts,
            replace=False, p=probs)
        test_pts = bss[test_pts]

        logger.debug('Comparing batch sizes: %s', test_pts)

    # here we could naively just loop over the test_pts, but this will be
    # a very inefficient use of our cores if we are on fast settings and
    # have many cores. Furthermore, some batch sizes will almost certainly
    # run faster than others. So alas, in the name of performance, this is
    # going to look a lot like _run_and_collate but dissimilar enough to not be
    # worth calling it

    folder = str(uuid.uuid4())
    os.makedirs(folder)

    loops = 0  # number spawned // test_pts.shape[0]
    last_loop_printed = 0
    cur_ind = 0  # in test_pts
    current_processes = []
    target_num_loops = max(
        settings.batch_pt_min_inits,
        cores // test_pts.shape[0]
    )
    while loops < target_num_loops:
        while len(current_processes) == cores:
            if last_loop_printed < loops:
                logger.debug('On loop %s/%s',
                             loops + 1, settings.batch_pt_min_inits)
                last_loop_printed = loops
            time.sleep(0.1)

            for i in range(len(current_processes) - 1, -1, -1):
                if not current_processes[i].is_alive():
                    current_processes.pop(i)

        fname = os.path.join(folder, f'{cur_ind}_{loops}.npz')
        bs = int(test_pts[cur_ind])
        proc = mp.Process(
            target=_train_with_perf,
            args=(
                model_loader, dataset_loader, loss_loader, fname,
                accuracy_style, bs, lr_min_over_batch * bs,
                lr_max_over_batch * bs, cycle_time_epochs,
                cycle_time_epochs, incl_raw
            )
        )
        proc.start()
        current_processes.append(proc)

        cur_ind += 1
        if cur_ind >= test_pts.shape[0]:
            cur_ind = 0
            loops += 1

    logger.debug('Waiting for %s currently running trials to end...',
                 len(current_processes))

    for proc in current_processes:
        proc.join()

    logger.debug('Organizing and interpreting batch size performance info...')

    all_final_perfs = np.zeros((test_pts.shape[0], loops))
    all_final_lse_perfs = np.zeros(test_pts.shape[0])

    raws_dict = dict()

    for i, bs in enumerate(test_pts):
        trials = []
        trials_raw = [] if incl_raw else None
        for trial in range(loops):
            fname = os.path.join(folder, f'{i}_{trial}.npz')
            with np.load(fname) as infile:
                final_perf = infile['final_perf']
                if np.isnan(final_perf).sum() > 0:
                    logger.debug('Found some nans, treating them as inf bad')
                    final_perf[np.isnan(final_perf)] = 0
                trials.append(final_perf)

                if incl_raw:
                    perf = infile['perf']
                    if np.isnan(perf).sum() > 0:
                        logger.debug('Found some nans in raw perfs')
                        perf[np.isnan(perf)] = 0
                    trials_raw.append(perf)
            os.remove(fname)
        trials = np.concatenate(trials)

        old_settings = np.seterr(under='ignore')
        lse_trials = scipy.special.logsumexp(trials)
        np.seterr(**old_settings)

        all_final_perfs[i] = trials
        all_final_lse_perfs[i] = lse_trials

        if incl_raw:
            trials_raw = np.stack(trials_raw)
            smoothed_trials_raw = scipy.signal.savgol_filter(
                trials_raw, smooth_window_size(trials_raw.shape[1]), 1
            )
            old_settings = np.seterr(under='ignore')
            lse_smoothed_trials_raw = scipy.special.logsumexp(
                smoothed_trials_raw, axis=0)
            np.seterr(**old_settings)

            raws_dict[f'raw_{bs}'] = trials_raw
            raws_dict[f'smoothed_raw_{bs}'] = smoothed_trials_raw
            raws_dict[f'lse_smoothed_raw_{bs}'] = lse_smoothed_trials_raw

    os.rmdir(folder)

    best_ind = np.argmax(all_final_lse_perfs)
    best_bs = int(test_pts[best_ind])

    np.savez_compressed(
        os.path.join(mainfolder, 'bs_sampled.npz'),
        bss=test_pts, final=all_final_perfs, lse_final=all_final_lse_perfs,
        **raws_dict
    )

    logger.info('Found best batch size of those tested: %s', best_bs)

    return best_bs, test_pts, loops

def tune(model_loader: typing.Tuple[str, str, tuple, dict],
         dataset_loader: typing.Tuple[str, str, tuple, dict],
         loss_loader: typing.Tuple[str, str, tuple, dict],
         accuracy_style: str,
         folder: str, cores: int,
         settings: HyperparameterSettings,
         store_up_to: AnalysisSettings,
         logger: logging.Logger = None):
    r"""Finds the optimal learning rate and batch size for the specified model
    on the specified dataset trained with the given loss. Stores the following
    information:

    .. code:: none

        folder/
            final.json
                {'lr_start': float, 'lr_end': float, 'batch_size': float,
                 'cycle_size_epochs': int, 'epochs': int}
            misc.json
                Variables that went into the final output. Typically selected
                via heuristics, constants, or come from the hyperparameter
                settings. Some may be deduced from the numpy array files
                directly

                {
                    'initial_batch_size': int,
                    'initial_cycle_time': int,
                    'initial_min_lr': float,
                    'initial_max_lr': float,
                    'initial_lr_num_to_val': int,
                    'initial_lr_num_trials': int,
                    'initial_lr_window_size': int,
                    'initial_lr_sweep_result_min': float,
                    'initial_lr_sweep_result_max': float,
                    'second_min_lr': float,
                    'second_max_lr': float
                }

            lr_vs_perf.npz
                lrs=np.ndarray[number of batches]
                perfs=np.ndarray[trials, number of batches]
                smoothed_perfs=np.ndarray[trials, number of batches]
                lse_smoothed_perfs=np.ndarray[trials, number of batches]
                perf_derivs=np.ndarray[trials, number_of_batches]
                smoothed_perf_derivs=np.ndarray[trials, number of batches]
                mean_smoothed_perf_derivs=np.ndarray[number of batches]
                lse_smoothed_perf_then_derivs=np.ndarray[number of batches]
                    lse = log sum exp. when there are many trials, the mean
                    gets overly pessimistic from bad initializations,
                    so LSE is more stable. however, we can't do lse on the
                    smoothed derivatives because then derivatives will tend
                    to be positive everywhere, so we have to smooth first,
                    then take lse, then take derivative
                lse_smoothed_perf_then_derivs_then_smooth=np.ndarray[
                    number of batches]. the smoothed variant of the pervious
                lr_range=np.ndarray[2]
                    min, max for the good range of learning rates
            bs_vs_perf.npz  (bs=batch_size)
                Where a single batch is tried multiple times, we take the
                mean over those times to ensure bss contains only unique
                values and hence can be treated like lrs

                bss=np.ndarray[number of batches]
                perfs=np.ndarray[trials, number of batches]
                smoothed_perfs=np.ndarray[number of batches]
                lse_smoothed_perfs=np.ndarray[number of batches]
                perf_derivs=np.ndarray[trials, number_of_batches]
                smoothed_perf_derivs=np.ndarray[trials, number of batches]
                mean_smoothed_perf_derivs=np.ndarray[number of batches]
                lse_smoothed_perf_then_derivs=np.ndarray[number of batches]
                bs_range=np.ndarray[2]
                    min, max for the good range of batch sizes
            lr_vs_perf2.npz
                only stored if settings.rescan_lr_after_bs. looks exactly
                like lr_vs_perf.npz, except these runs are performed with
                the newly selected batch size
            bs_sampled.npz
                only stored if settings.batch_pts > 0

                bss=np.ndarray[num bs attempted]
                final=np.ndarray[num bs attempted, trials]
                    final performance for batch size i for each trial
                lse_final=np.ndarray[num bs attempted]
                    final logsumexp performance for each batch size, argmax
                    is the selected batch size. If you want this to nicely
                    be below the maximum, subtract log(trials) and note
                    this does not effect the argmax

                raw_i=np.ndarray[trials, number of batches]
                    only if store_up_to.hparam_selection_specific_imgs,
                    same for the *_raw_i

                    i is a sampled batch size and raw_i[t, j] is the
                    performance of the model after iteration j for
                    batch size i on trial t.
                smoothed_raw_i=np.ndarray[trials, number of batches]
                lse_smoothed_raw_i=np.ndarray[number of batches]


    :param model_loader: describes which module and corresponding attribute can
        be passed what arguments and keyword arguments to produce the
        nn.Module with a random initialization which can be trained

        .. code::python

            model_loader = ('torch.nn', 'Linear', tuple(20, 10),
                            {'bias': True})

    :param dataset_loader: describes which module and corresponding attribute
        can be passed what arguments and keyword arguments to produce the
        training dataset and validation dataset.
    :param loss_loader: describes which module and corresponding attribute can
        be passed what arguments and keyword arguments to produce the nn.Module
        that converts (y_pred, y) to a scalar which should be minimized
    :param folder: where to save the output to
    :param cores: how many cores to use; 1 for just the main process
    :param settings: the settings to use to tune the learning rate and batch
        size
    :param store_up_to: the information stored should be at least what is
        required to produce this analysis
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    os.makedirs(folder)

    train_set, _ = ignite_simple.utils.invoke(dataset_loader)

    logger.info('Performing initial learning rate sweep...')
    init_batch_size = 64
    init_cycle_time = int(np.clip(150000 // len(train_set), 2, 5) * 2)
    if isinstance(settings.warmup_pts, float):
        warmup_pts = int(len(train_set) * settings.warmup_pts)
    else:
        warmup_pts = settings.warmup_pts
    warmup_batch = settings.warmup_batch
    warmup_lr = settings.warmup_lr

    lr_min, lr_max, lr_initial_window_size, lr_initial_trials = (
        _select_lr_from(
            model_loader, dataset_loader, loss_loader, accuracy_style,
            os.path.join(folder, 'lr_vs_perf.npz'), cores, settings,
            store_up_to, logger, init_cycle_time, init_batch_size,
            settings.lr_start, settings.lr_end, settings.lr_width_only_gradients,
            warmup_lr, warmup_batch, warmup_pts
        )
    )
    initial_lr_sweep_result_min, initial_lr_sweep_result_max = lr_min, lr_max
    initial_lr_num_to_val = min(NUM_TO_VAL_MAX, len(train_set))

    logger.info('Performing initial batch size sweep...')

    # The trick is the increasing the batch size requires a corresponding
    # increase in learning rate. We don't want to include the lr range
    # except insofar as taking that into account as otherwise these
    # results would be even muddier than they already are

    lr_avg_over_batch = ((lr_min + lr_max) / 2) / init_batch_size

    bs_sweep_lr_min = lr_avg_over_batch * settings.batch_start
    bs_sweep_lr_max = lr_avg_over_batch * settings.batch_end

    result = _run_and_collate(
        _batch_vs_perf, {
            'model_loader': model_loader,
            'dataset_loader': dataset_loader,
            'loss_loader': loss_loader,
            'accuracy_style': accuracy_style,
            'batch_start': settings.batch_start,
            'batch_end': settings.batch_end,
            'lr_start': bs_sweep_lr_min,
            'lr_end': bs_sweep_lr_max,
            'cycle_time_epochs': init_cycle_time
        }, cores, settings.batch_rn_min_inits
    )

    logger.debug('Organizing and interpreting batch size sweep...')

    bss = result['bss'][0]
    bs_perfs = result['perfs']
    if np.sum(np.isnan(bs_perfs)) > 0:
        logger.debug('Batch size sweep exploded on some initializations')
        logger.debug('Forcibly enabling second LR sweep')
        settings.rescan_lr_after_bs = True

        bs_perfs[np.isnan(bs_perfs)] = 0

    bs_sweep_trials = int(bs_perfs.shape[0])

    window_size = smooth_window_size(bs_perfs.shape[1])

    smoothed_bs_perf = scipy.signal.savgol_filter(
        bs_perfs, window_size, 1
    )
    old_settings = np.seterr(under='ignore')
    lse_smoothed_bs_perf = scipy.special.logsumexp(
        smoothed_bs_perf, axis=0
    )
    np.seterr(**old_settings)
    lse_smoothed_bs_perf_then_derivs = np.gradient(
        lse_smoothed_bs_perf, axis=0)

    bs_perf_derivs = np.gradient(bs_perfs, axis=-1)
    smoothed_bs_perf_derivs = scipy.signal.savgol_filter(
        bs_perfs, window_size, 1, deriv=1)

    mean_smoothed_bs_perf_derivs = smoothed_bs_perf_derivs.mean(0)

    bs_min, bs_max = find_with_derivs(bss, lse_smoothed_bs_perf_then_derivs)
    bs_min, bs_max = int(bs_min), int(bs_max)

    logger.info('Batch size range: [%s, %s) (found from %s trials)',
                bs_min, bs_max, bs_perfs.shape[0])

    np.savez_compressed(
        os.path.join(folder, 'bs_vs_perf.npz'),
        bss=bss, perfs=bs_perfs,
        perf_derivs=bs_perf_derivs,
        smoothed_perfs=smoothed_bs_perf,
        smoothed_perf_derivs=smoothed_bs_perf_derivs,
        mean_smoothed_perf_derivs=mean_smoothed_bs_perf_derivs,
        lse_smoothed_perf_then_derivs=lse_smoothed_bs_perf_then_derivs,
        bs_range=np.array([bs_min, bs_max]))

    if settings.batch_pts > 1:
        batch_size, batch_pts_checked, num_batch_loops = _select_batch_size_from(
            model_loader, dataset_loader, loss_loader, accuracy_style, folder,
            cores, settings, store_up_to, logger, init_cycle_time, bss,
            lse_smoothed_bs_perf_then_derivs, bs_min, bs_max,
            lr_min / init_batch_size, lr_max / init_batch_size)
    else:
        batch_size = (bs_min + bs_max) // 2
        batch_pts_checked = []
        num_batch_loops = -1
        logger.info('Choosing mean batch size: %s', batch_size)

    if settings.rescan_lr_after_bs and batch_size != init_batch_size:
        logger.info('Finding learning rate range on new batch size...')
        second_min_lr = (settings.lr_start / init_batch_size) * batch_size
        second_max_lr = (settings.lr_end / init_batch_size) * batch_size
        lr_min, lr_max, second_lr_window_size, second_lr_num_trials = _select_lr_from(
            model_loader, dataset_loader, loss_loader, accuracy_style,
            os.path.join(folder, 'lr_vs_perf2.npz'), cores, settings,
            store_up_to, logger, init_cycle_time, init_batch_size,
            second_min_lr, second_max_lr, settings.lr_width_only_gradients,
            warmup_lr, warmup_batch, warmup_pts
        )
    else:
        second_min_lr = float('nan')
        second_max_lr = float('nan')
        second_lr_window_size = float('nan')
        second_lr_num_trials = float('nan')
        lr_min = (lr_min / init_batch_size) * batch_size
        lr_max = (lr_max / init_batch_size) * batch_size

    with open(os.path.join(folder, 'final.json'), 'w') as outfile:
        json.dump({'lr_start': lr_min, 'lr_end': lr_max,
                   'batch_size': batch_size,
                   'cycle_size_epochs': init_cycle_time,
                   'epochs': init_cycle_time * 4}, outfile)

    with open(os.path.join(folder, 'misc.json'), 'w') as outfile:
        json.dump(
            {
                'initial_batch_size': init_batch_size,
                'initial_cycle_time': init_cycle_time,
                'initial_min_lr': settings.lr_start,
                'initial_max_lr': settings.lr_end,
                'initial_lr_num_to_val': initial_lr_num_to_val,
                'initial_lr_num_trials': lr_initial_trials,
                'initial_lr_window_size': lr_initial_window_size,
                'initial_lr_sweep_result_min': initial_lr_sweep_result_min,
                'initial_lr_sweep_result_max': initial_lr_sweep_result_max,
                'initial_avg_lr': (initial_lr_sweep_result_min + initial_lr_sweep_result_max) / 2,
                'initial_min_batch': settings.batch_start,
                'initial_max_batch': settings.batch_end,
                'initial_batch_num_to_val': initial_lr_num_to_val,
                'initial_batch_num_trials': bs_sweep_trials,
                'batch_sweep_result_min': bs_min,
                'batch_sweep_result_max': bs_max,
                'batch_sweep_result': batch_size,
                'batch_sweep_num_pts': len(batch_pts_checked),
                'batch_sweep_pts_list': list(int(i) for i in batch_pts_checked),
                'batch_sweep_trials_each': num_batch_loops,
                'second_min_lr': second_min_lr,
                'second_max_lr': second_max_lr,
                'second_lr_num_trials': second_lr_num_trials,
                'second_lr_window_size': second_lr_window_size,
                'lr_sweep_result_min': lr_min,
                'lr_sweep_result_max': lr_max,
                'warmup_pts': warmup_pts,
                'warmup_lr': warmup_lr,
                'warmup_batch': warmup_batch,
                'lr_width_only_gradients': settings.lr_width_only_gradients
            },
            outfile
        )

    logger.debug('Tuning completed successfully')
