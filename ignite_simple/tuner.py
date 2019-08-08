"""This module is responsible for tuning the learning rate and batch size for
training a module."""
import ignite_simple  # pylint: disable=unused-import
import typing
import importlib
from ignite_simple.hyperparams import HyperparameterSettings
from ignite_simple.analarams import AnalysisSettings
from ignite_simple.vary_bs_loader import BatchSizeVaryingDataLoader
from ignite_simple.range_finder import smooth_window_size, find_with_derivs
import ignite_simple.trainer
import torch
import torch.utils.data as data
import numpy as np
from ignite.engine import Events
import os
import uuid
import multiprocessing as mp
import logging
import time
import scipy.signal
import scipy.special
import json

def _invoke(loader):
    modulename, attrname, args, kwargs = loader

    module = importlib.import_module(modulename)
    return getattr(module, attrname)(*args, **kwargs)

def _valldr(val_set, num_to_val):
    if num_to_val == len(val_set):
        return data.DataLoader(val_set, batch_size=64 * 3)

    valinds = np.random.choice(len(val_set), num_to_val, replace=False)
    valinds = torch.from_numpy(valinds).long()
    return data.DataLoader(data.Subset(val_set, valinds),
                           batch_size=min(64 * 3, num_to_val))

def _task_loader(dataset_loader, batch_size, shuffle, drop_last):
    train_set, val_set = _invoke(dataset_loader)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_set, val_set, train_loader

def _store_lr_and_perf(lrs, perfs, cur_iter, num_to_val, tnr,
                       state: ignite_simple.trainer.TrainState):
    valldr = _valldr(state.val_set, num_to_val)
    state.evaluator.run(valldr)

    lrs[cur_iter[0]] = state.lr_scheduler.get_param()
    perfs[cur_iter[0]] = state.evaluator.state.metrics['perf']

def _increment(cur, tnr, state):
    cur[0] += 1


def _lr_vs_perf(model_loader, dataset_loader, loss_loader, outfile,
                accuracy_style, lr_start, lr_end, batch_size,
                cycle_time_epochs):
    train_set, val_set = _invoke(dataset_loader)

    num_train_iters = (len(train_set) // batch_size) * (cycle_time_epochs // 2)

    cur_iter = [0]
    num_to_val = min(64 * 3, len(val_set))

    lrs = np.zeros(num_train_iters)
    perfs = np.zeros(num_train_iters)

    tnr_settings = ignite_simple.trainer.TrainSettings(
        accuracy_style, model_loader, loss_loader,
        (__name__, '_task_loader',
         (dataset_loader, batch_size, True, True), dict()),
        (
            (Events.ITERATION_COMPLETED,
             (__name__, '_store_lr_and_perf',
              (lrs, perfs, cur_iter, num_to_val), dict())),
            (Events.ITERATION_COMPLETED,
             (__name__, '_increment', (cur_iter,), dict()))
        ),
        lr_start,
        lr_end,
        cycle_time_epochs,
        cycle_time_epochs // 2
    )
    ignite_simple.trainer.train(tnr_settings)
    np.savez_compressed(outfile, lrs=lrs, perfs=perfs)

def _task_loader_bs(dataset_loader, batch_start, batch_end, epochs):
    train_set, val_set = _invoke(dataset_loader)
    train_loader = BatchSizeVaryingDataLoader(
        train_set, batch_start, batch_end, epochs)
    return train_set, val_set, train_loader


def _store_bs_and_perf(bss, perfs, cur, num_to_val, tnr,
                       state: ignite_simple.trainer.TrainState):
    valldr = _valldr(state.val_set, num_to_val)
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
    train_set, val_set = _invoke(dataset_loader)

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
    num_to_val = min(64 * 3, len(val_set))

    tnr_settings = ignite_simple.trainer.TrainSettings(
        accuracy_style, model_loader, loss_loader,
        (__name__, '_task_loader_bs',
         (dataset_loader, batch_start, batch_end, epochs), dict()),
        (
            (Events.ITERATION_COMPLETED,
             (__name__, '_store_bs_and_perf',
              (bss, perfs, cur, num_to_val), dict())),
            (Events.COMPLETED,
             (__name__, '_store_last_bs', (perfs, cur), dict()))
        ),
        lr_start,
        lr_end,
        2,
        1
    )
    ignite_simple.trainer.train(tnr_settings)
    np.savez_compressed(outfile, bss=bss, perfs=perfs)

def _store_perf(perfs, cur, num_to_val, tnr, state):
    valldr = _valldr(state.val_set, num_to_val)
    state.evaluator.run(valldr)
    perfs[cur[0]] = state.evaluator.state.metrics['perf']

def _train_with_perf(model_loader, dataset_loader, loss_loader, outfile,
                     accuracy_style, batch_size, lr_start, lr_end,
                     cycle_time_epochs, epochs, with_raw):
    train_set, val_set = _invoke(dataset_loader)

    final_perf = np.zeros(1)
    final_ind = [0]
    handlers = [
        (Events.COMPLETED,
         (__name__, '_store_perf',
          (final_perf, final_ind, len(val_set)), dict()))
    ]
    if with_raw:
        num_iters = (len(train_set) // batch_size) * epochs
        num_to_val = min(64 * 3, len(val_set))
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
        handlers, lr_start, lr_end, cycle_time_epochs, epochs
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
                    batch_size) -> typing.Tuple[int, int]:
    result = _run_and_collate(
        _lr_vs_perf, {
            'model_loader': model_loader,
            'dataset_loader': dataset_loader,
            'loss_loader': loss_loader,
            'accuracy_style': accuracy_style,
            'lr_start': settings.lr_start,
            'lr_end': settings.lr_end,
            'batch_size': batch_size,
            'cycle_time_epochs': cycle_time_epochs
        }, cores, settings.lr_min_inits
    )

    logger.debug('Organizing and interpreting learning rate sweep...')

    lrs = result['lrs'][0]
    lr_perfs = result['perfs']
    window_size = smooth_window_size(lrs.shape[0])

    lr_perf_derivs = np.gradient(lr_perfs, axis=-1)
    smoothed_lr_perf_derivs = scipy.signal.savgol_filter(
        lr_perfs, window_size, 1, deriv=1)
    mean_smoothed_lr_perf_derivs = smoothed_lr_perf_derivs.mean(0)

    lr_min, lr_max = find_with_derivs(lrs, mean_smoothed_lr_perf_derivs)

    np.savez_compressed(
        outfile, lrs=lrs, perfs=lr_perfs,
        perf_derivs=lr_perf_derivs,
        smoothed_perf_derivs=smoothed_lr_perf_derivs,
        mean_smoothed_perf_derivs=mean_smoothed_lr_perf_derivs,
        lr_range=np.array([lr_min, lr_max]))

    logger.info('Learning rate range: [%s, %s)', lr_min, lr_max)
    return lr_min, lr_max


def _select_batch_size_from(model_loader, dataset_loader, loss_loader,
                            accuracy_style, mainfolder, cores, settings,
                            store_up_to, logger, cycle_time_epochs, bss,
                            mean_smoothed_bs_perf_derivs,
                            bs_min, bs_max, lr_min_over_batch,
                            lr_max_over_batch) -> int:
    settings: HyperparameterSettings
    store_up_to: AnalysisSettings

    bs_min_ind = int((bss == bs_min).argmax())
    bs_max_ind = int((bss == bs_max).argmax())

    incl_raw = store_up_to.hparam_selection_specific_imgs

    if bs_min == bs_max:
        logger.info('Only found a single good batch size, using that without '
                    + 'further investigation')
        return bs_min

    if bs_max - bs_min <= settings.batch_pts:
        logger.debug('Found %s good batch sizes and willing to try up to %s, '
                     + 'so testing all of them.', bs_max - bs_min,
                     settings.batch_pts)
        test_pts = np.arange(bs_min, bs_max)
    else:
        probs = mean_smoothed_bs_perf_derivs[bs_min_ind:bs_max_ind]
        probs = scipy.special.softmax(probs)

        test_pts = np.random.choice(
            np.arange(bs_min, bs_max), settings.batch_pts,
            replace=False, p=probs)

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
    cur_ind = 0  # in test_pts
    current_processes = []
    while loops < settings.batch_pt_min_inits:
        while len(current_processes) == cores:
            time.sleep(0.1)

            for i in range(len(current_processes) - 1, -1, -1):
                if not current_processes[i].is_alive():
                    current_processes.pop(i)

        fname = os.path.join(folder, f'{cur_ind}_{loops}.npz')
        bs = int(test_pts[cur_ind])
        logger.debug('Starting run with batch size %s', bs)
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
                trials.append(infile['final_perf'])
                if incl_raw:
                    trials_raw.append(infile['perf'])
            os.remove(fname)
        trials = np.concatenate(trials)
        lse_trials = scipy.special.logsumexp(trials)

        all_final_perfs[i] = trials
        all_final_lse_perfs[i] = lse_trials

        if incl_raw:
            trials_raw = np.stack(trials_raw)
            smoothed_trials_raw = scipy.signal.savgol_filter(
                trials_raw, smooth_window_size(trials_raw.shape[1]), 1
            )
            lse_smoothed_trials_raw = scipy.special.logsumexp(
                smoothed_trials_raw, axis=0)

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

    return best_bs

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

    .. code::

        folder/
            final.json
                {'lr_start': float, 'lr_end': float, 'batch_size': float}
            lr_vs_perf.npz
                lrs=np.ndarray[number of batches]
                perfs=np.ndarray[trials, number of batches]
                perf_derivs=np.ndarray[trials, number_of_batches]
                smoothed_perf_derivs=np.ndarray[trials, number of batches]
                mean_smoothed_perf_derivs=np.ndarray[number of batches]
                lr_range=np.ndarray[2]
                    min, max for the good range of learning rates
            bs_vs_perf.npz  (bs=batch_size)
                Where a single batch is tried multiple times, we take the
                mean over those times to ensure bss contains only unique
                values and hence can be treated like lrs

                bss=np.ndarray[number of batches]
                perfs=np.ndarray[trials, number of batches]
                perf_derivs=np.ndarray[trials, number_of_batches]
                smoothed_perf_derivs=np.ndarray[trials, number of batches]
                mean_smoothed_perf_derivs=np.ndarray[number of batches]
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

    train_set, _ = _invoke(dataset_loader)

    logger.info('Performing initial learning rate sweep...')
    init_batch_size = 64
    init_cycle_time = int(np.clip(150000 // len(train_set), 2, 5) * 2)

    lr_min, lr_max = _select_lr_from(
        model_loader, dataset_loader, loss_loader, accuracy_style,
        os.path.join(folder, 'lr_vs_perf.npz'), cores, settings,
        store_up_to, logger, init_cycle_time, init_batch_size
    )

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
    window_size = smooth_window_size(bss.shape[0])

    bs_perf_derivs = np.gradient(bs_perfs, axis=-1)
    smoothed_bs_perf_derivs = scipy.signal.savgol_filter(
        bs_perfs, window_size, 1, deriv=1)
    mean_smoothed_bs_perf_derivs = smoothed_bs_perf_derivs.mean(0)

    bs_min, bs_max = find_with_derivs(bss, mean_smoothed_bs_perf_derivs)
    bs_min, bs_max = int(bs_min), int(bs_max)

    logger.info('Batch size range: [%s, %s)', bs_min, bs_max)

    np.savez_compressed(
        os.path.join(folder, 'bs_vs_perf.npz'), bss=bss, perfs=bs_perfs,
        perf_derivs=bs_perf_derivs,
        smoothed_perf_derivs=smoothed_bs_perf_derivs,
        mean_smoothed_perf_derivs=mean_smoothed_bs_perf_derivs,
        bs_range=np.array([bs_min, bs_max]))

    if settings.batch_pts > 1:
        batch_size = _select_batch_size_from(
            model_loader, dataset_loader, loss_loader, accuracy_style, folder,
            cores, settings, store_up_to, logger, init_cycle_time, bss,
            mean_smoothed_bs_perf_derivs, bs_min, bs_max,
            lr_min / init_batch_size, lr_max / init_batch_size)
    else:
        batch_size = (bs_min + bs_max) // 2
        logger.info('Choosing mean batch size: %s', batch_size)

    if settings.rescan_lr_after_bs and batch_size != init_batch_size:
        logger.info('Finding learning rate range on new batch size...')
        lr_min, lr_max = _select_lr_from(
            model_loader, dataset_loader, loss_loader, accuracy_style,
            os.path.join(folder, 'lr_vs_perf2.npz'), cores, settings,
            store_up_to, logger, init_cycle_time, init_batch_size
        )

    with open(os.path.join(folder, 'final.json'), 'w') as outfile:
        json.dump({'lr_start': lr_min, 'lr_end': lr_max,
                   'batch_size': batch_size}, outfile)