"""Handles performing the model parameter sweep."""

import typing
import ignite_simple.gen_sweep.param_selectors as ps
import ignite_simple.hyperparams as hyperparams
import ignite_simple.dispatcher as disp
import ignite_simple.utils as mutils
import psutil
import os
import pickle
import json
import importlib
import numpy as np
from sortedcontainers import SortedKeyList
from functools import partial
import csv
import logging

logger = logging.getLogger(__name__)

def _post_hparam_sweep(listeners: typing.Iterable[ps.SweepListener],
                       params: tuple, mm_folder: str):
    filen = os.path.join(mm_folder, 'hparams', 'final.json')
    if not os.path.exists(filen):
        logger = logging.getLogger(__name__)
        logger.error('Expected file %s to exist, corresponding with hparams'
                     + ' for params %s, but it does not', filen, params)
        raise FileNotFoundError(filen)

    with open(filen, 'r') as infile:
        final = json.load(infile)

    lr_min = final['lr_start']
    lr_max = final['lr_end']
    bs = final['batch_size']

    for lst in listeners:
        lst: ps.SweepListener
        lst.on_hparams_found(params, lr_min, lr_max, bs)


def _post_trials(listeners: typing.Iterable[ps.SweepListener],
                 params: tuple, mm_folder: str, start_at_trial_ind: int):
    filen = os.path.join(mm_folder, 'results.npz')
    if not os.path.exists(filen):
        logger = logging.getLogger(__name__)
        logger.error('Expected file %s to exist, corresponding with results'
                     + ' for params %s, but it does not', filen, params)
        raise FileNotFoundError(filen)

    with np.load(filen) as infile:
        l_train = infile['final_loss_train'][start_at_trial_ind:]
        p_train = infile['final_perf_train'][start_at_trial_ind:]
        l_val = infile['final_loss_val'][start_at_trial_ind:]
        p_val = infile['final_perf_val'][start_at_trial_ind:]

    for lst in listeners:
        lst: ps.SweepListener
        lst.on_trials_completed(params, p_train, l_train, p_val, l_val)

class ParamsTask:
    """Describes a task to call model_manager.train for a particular set of
    model parameters. This isn't actually a disp.Task, although it can be
    converted to one.

    :ivar tuple params: the parameters. this is the instance variable used
        for equality checks.
    :ivar int trials: the number of trials to perform.
    """
    def __init__(self, params: tuple, trials: int) -> None:
        self.params = tuple(params)
        self.trials = trials

    def copy(self) -> 'ParamsTask':
        return ParamsTask(self.params, self.trials)

    def as_task(self, module_name: str,
                hparams: hyperparams.HyperparameterSettings,
                folder: str, iden: int, trials: int,
                sweep_cores: int,
                listeners: typing.Iterable[ps.SweepListener],
                ntrials_disp: int) -> disp.Task:
        """Creates the task object which performs the specified number of
        trials with this tasks parameters, assuming that the directory
        structure is as follows:

        folder/points/
            <id>/
                <result from model manager>

        This function assumes that we always perform the hparam sweep in a
        separate call to the trial sweep for the purposes of handling the
        listeners.
        """
        md = importlib.import_module(module_name)
        acc_style = getattr(md, 'accuracy_style')
        mm_folder = os.path.join(folder, 'points', str(iden))
        if trials <= 0:
            callback = partial(_post_hparam_sweep, listeners, self.params, mm_folder)
        else:
            callback = partial(_post_trials, listeners, self.params, mm_folder,
                               ntrials_disp)
        return disp.Task(
            'ignite_simple.model_manager', 'train',
            (
                (module_name, 'model', self.params, dict()),
                (module_name, 'dataset', tuple(), dict()),
                (module_name, 'loss', tuple(), dict()),
                mm_folder,
                hparams,
                'none',
                'none',
                acc_style,
                max(trials, 0),
                True,
                os.path.join(folder, 'points', 'history', str(iden)),
                sweep_cores if trials <= 0 else trials,
                True
            ),
            dict(),
            sweep_cores if trials <= 0 else trials,
            callback
        )

class ParamsTaskQueue(disp.TaskQueue, ps.SweepListener):
    """This task queue acts like a greedy task queue, except it only works for
    tasks which amount to calling model_manager.train for a particular set of
    models.

    The process for selecting tasks is as follows:

        - If we have a hparameter sweep queued and we have enough cores to do
          one, do that.
        - Perform as many trials as possible. Note we can only perform trials
          on parameters we already know the hyperparameters for

    :ivar int total_cores: the number of physical cores that are available
    :ivar int sweep_cores: the number of cores to use for sweeping (not greater
        than the number of total cores)
    :ivar str module: the module which we are getting the model/dataset/etc
    :ivar HyperparameterSettings hparams: strategy for hyperparameters
    :ivar str folder: the folder containing the points folder
    :ivar list[SweepListener] listeners: the sweep listeners. contains self.
    :ivar set[tuple] in_progress: parameters which are currently in progress.
    :ivar list[ParamsTask] sweeps: the sweeps that need to be performed in an
        arbitrary order
    :ivar dict[tuple, int] params_to_sweeps_ind: a lookup that goes from the
        parameters of tasks to the index in sweeps if a sweep is still
        necessary.
    :ivar SortedList[ParamsTask] trials: the trials that need to be performed,
        in ascending order of the number of trials.
    :ivar dict[tuple, int] params_to_id: dictionary which converts a given
        set of parameters to the corresponding unique identifier for that set
        of parameters.
    :ivar dict[tuple, int] params_to_ntrials: dictionary which converts a
        given set of parameters to the corresponding number of trials that have
        been dispatched for those parameters
    :ivar int next_id: the next id that should be given out to a set of
        parameters and then incremented.
    :ivar dict[tuple, ParamsTask] params_to_task: a lookup that goes from
        params lists to param tasks, where this only contains tasks which have
        not yet been dispatched, and does not include tasks which are in sweeps
    :ivar int _len: number of actual tasks currently in queue
    :ivar bool expecting_more_trials: True to prevent saying we are out of
        trials, False otherwise
    """
    def __init__(self, total_cores: int, sweep_cores: int, module: str,
                 hparams: hyperparams.HyperparameterSettings,
                 folder: str,
                 listeners: typing.List[ps.SweepListener]):
        self.total_cores = total_cores
        self.sweep_cores = sweep_cores
        self.module = module
        self.hparams = hparams
        self.folder = folder
        self.listeners = list(listeners)
        self.listeners.append(self)
        self.in_progress = set()
        self.sweeps = list()
        self.params_to_sweeps_ind = dict()
        self.trials = SortedKeyList(key=lambda tsk: tsk.trials)
        self.params_to_id = dict()
        self.params_to_ntrials = dict()
        self.next_id = 0
        self.params_to_task = dict()
        self._len = 0
        self.expecting_more_trials = False

    def add_task_by_params(self, params: tuple, trials: int) -> None:
        """Adds a task to this queue based on the parameters which should
        be swept and the number of trials to perform. Regardless of the
        value for trials, this will ensure that the hyperparameters for
        the given model parameters have been found.
        """
        sweep_id = self.params_to_id.get(params)
        if sweep_id is None:
            sweep_id = self.next_id
            self.next_id += 1
            self.sweeps.append(ParamsTask(params, 0))
            self.params_to_sweeps_ind[params] = len(self.sweeps) - 1
            self.params_to_id[params] = sweep_id
            self.params_to_ntrials[params] = 0
            self._len += 1

        if trials <= 0:
            return

        tsk = self.params_to_task.get(params)
        if tsk is not None:
            self.trials.remove(tsk)
            tsk.trials += trials
            self.trials.add(tsk)
            return

        tsk = ParamsTask(params, trials)
        self.params_to_task[params] = tsk
        self.trials.add(tsk)
        self._len += 1

    def set_total_cores(self, total_cores):
        self.total_cores = total_cores

    def on_hparams_found(self, values, lr_min, lr_max, batch_size):
        logger.debug('Found hyperparameters for %s: lr=(%s, %s), bs=%s',
            values, lr_min, lr_max, batch_size)
        self.in_progress.remove(values)

    def on_trials_completed(self, values, perfs_train, losses_train, perfs_val, losses_val):
        logger.info('Completed some trials for %s - mean train/val perf = %s / %s',
            values, perfs_train.mean(), perfs_val.mean())
        logger.debug('%s - perf: %s, loss: %s, val - perf: %s, loss: %s',
            values, perfs_train, losses_train, perfs_val, losses_val)
        self.in_progress.remove(values)
        self.params_to_ntrials[values] += len(losses_train)

    def _get_next_task(self, cores):
        # Pseudocode:
        #   if we have enough cores to sweep and a sweep available then
        #       pop from the end of sweeps (so only one index changes)
        #       remove from params_to_sweeps_ind
        #       add to in_progress
        #       return
        #
        #   pop the item with the most number of trials from trials, ignoring
        #   ones which are already in progress or haven't been sweep yet
        #
        #   if we cannot finish this then
        #       build the disp.Task which does the right # of trials
        #       update the remaining number of trials for this trial
        #       add to in_progress
        #       return the built disp.Task
        #   build the disp.Task which finishes the queued trials for this set
        #   remove from params_to_task
        #   add to in_progress
        #   return built disp.Task
        if cores <= 0:
            return None

        if cores >= self.sweep_cores and self.sweeps:
            swp: ParamsTask = self.sweeps.pop()
            del self.params_to_sweeps_ind[swp.params]
            self._len -= 1
            self.in_progress.add(swp.params)
            return swp.as_task(
                self.module, self.hparams, self.folder,
                self.params_to_id[swp.params],
                0, self.sweep_cores,
                self.listeners,
                self.params_to_ntrials[swp.params]
            )

        if not self.trials:
            return None

        pop_ind = len(self.trials) - 1
        while True:
            trl = self.trials[pop_ind]
            if (trl.params not in self.in_progress
                    and trl.params not in self.params_to_sweeps_ind):
                break

            pop_ind -= 1
            if pop_ind < 0:
                return None

        trl = self.trials.pop(pop_ind)
        if trl.trials > cores:
            trl.trials -= cores
            self.trials.add(trl)
            self.in_progress.add(trl.params)
            return trl.as_task(
                self.module, self.hparams, self.folder,
                self.params_to_id[trl.params],
                cores, self.sweep_cores,
                self.listeners,
                self.params_to_ntrials[trl.params]
            )
        del self.params_to_task[trl.params]
        self._len -= 1
        self.in_progress.add(trl.params)
        return trl.as_task(
            self.module, self.hparams, self.folder,
            self.params_to_id[trl.params],
            trl.trials, self.sweep_cores,
            self.listeners,
            self.params_to_ntrials[trl.params]
        )

    def get_next_task(self, cores):
        res = self._get_next_task(cores)
        if res is not None:
            logger.debug('starting task %s', str(res))
        return res

    def have_more_tasks(self):
        return (self.expecting_more_trials or self._len > 0)

    def __len__(self) -> int:
        return self._len


class Sweeper:
    """The instance of the class which performs all the individual operations
    required for sweeping over an arbitrary number of architectural
    parameters. Everything within this class occurs on the main thread.

    :ivar module_name: the module which contains the model, dataset, loss,
        and accuracy style. The model function has N parameters, where
        N is the number of parameters the param selector gives us.
    :ivar ParamSelector param_selector: the object which selects which
        combination of parameters to test
    :ivar ParamsTaskQueue tasks: the tasks that we know we are ready to
        perform. These are being immediately dispatched to other threads
        according to the number of cores required.
    :ivar int sweep_cores: the number of actual physical cores that we assign
        to sweeping. Note that this differs from the argument to sweep.
    :ivar str folder: the path to the folder we are saving things in.
        we actually store stuff at folder/points for the most part.
    :ivar HyperparameterSettings hparams: the hyperparameter settings used for
        hyperparameter tuning.
    """
    def __init__(self, module_name: str, param_selector: ps.ParamSelector,
                 tasks: ParamsTaskQueue, sweep_cores: int,
                 folder: str, hparams: hyperparams.HyperparameterSettings):
        self.module_name = module_name
        self.param_selector = param_selector
        self.tasks = tasks
        self.sweep_cores = sweep_cores
        self.folder = folder
        self.hparams = hparams

    def add_tasks(self, limit=100) -> None:
        """Adds as many tasks to the TaskQueue as the parameter selector will
        provide, without causing the tasks list to exceed the given limit. This
        will also update tasks.expecting_more_tasks.

        This must ensure that we don't enqueue multiple tasks which have the
        same hyperparameters, however we can update tasks to add more trials.
        """
        self.tasks.expecting_more_trials = True
        while (self.param_selector.has_more_trials()
               and len(self.tasks) < limit):
            res = self.param_selector.get_next_trials()
            if res is None:
                return
            prms, trls = res
            self.tasks.add_task_by_params(prms, trls)
        self.tasks.expecting_more_trials = False

    def store_params_lookup(self) -> None:
        """Stores the parameters that corresponded to each of the arbitrarily
        named folders in folder/points. This only works when we just swept
        and hence the details are still in memory.

        Produces the following files:

        folder/points/
            params_lookup.pt
                Contains a pickled tuple of two dictionaries. The first
                dictionary goes form parameter tuples to corresponding ids
                (names of folders). The second goes from ids to parameter
                tuples.
            params.csv
                A human-readable variant of params_lookup. A CSV file where
                the rows are <id>,<params>. Each individual parameter is given
                its own column. The first row is used for descriptions and
                should be skipped when parsing.
        """
        params_to_id = self.tasks.params_to_id
        ids_to_param = dict((v, k) for k, v in params_to_id.items())

        pickled_file = os.path.join(self.folder, 'points', 'params_lookup.pt')
        with open(pickled_file, 'wb') as outfile:
            pickle.dump((params_to_id, ids_to_param), outfile)

        csv_filen = os.path.join(self.folder, 'points', 'params.csv')
        with open(csv_filen, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Identifier / Folder Name', 'Parameters (one per column)'])
            for iden, params in ids_to_param.items():
                row = [iden]
                row.extend(params)
                row = [str(s) for s in row]
                csv_writer.writerow(row)


    def collate_files(self) -> None:
        """Collates the information found in each of the sweeps into locations
        which are more readily accessible. This only requires that the folder
        is set correctly and store_params_lookup has been called.

        Produces the following files:

        folder/points/
            funcres.pt
                Contains a pickled list of tuples of the following form:

                (params, min lr, max lr, bs, perfs_train, losses_train,
                 perfs_val, losses_val)
        """
        pickled_file = os.path.join(self.folder, 'points', 'params_lookup.pt')
        with open(pickled_file, 'rb') as infile:
            params_to_id, ids_to_param = pickle.load(infile)

        res = []
        for iden, params in ids_to_param.items():
            pfolder = os.path.join(self.folder, 'points', str(iden))
            hpfile = os.path.join(pfolder, 'hparams', 'final.json')
            with open(hpfile, 'r') as infile:
                hps = json.load(infile)

            lr_min = hps['lr_start']
            lr_max = hps['lr_end']
            bs = hps['batch_size']

            resfile = os.path.join(pfolder, 'results.npz')
            with np.load(resfile) as infile:
                l_train = infile['final_loss_train']
                p_train = infile['final_perf_train']
                l_val = infile['final_loss_val']
                p_val = infile['final_perf_val']

            res.append((params, lr_min, lr_max, bs, p_train, l_train,
                        p_val, l_val))

        fres_file = os.path.join(self.folder, 'points', 'funcres.pt')
        with open(fres_file, 'wb') as outfile:
            pickle.dump(res, outfile)

def sweep(module_name: str,
          param_selector: ps.ParamSelector,
          sweep_cores: int,
          hparams: typing.Union[str, hyperparams.HyperparameterSettings],
          folder: str) -> list:
    """Performs the given architectural parameter search on the problem defined
    in the given module.

    :param str module_name: the path to the module which contains the
        model / dataset / loss / accuracy style, where the model accepts
        the parameters which are being swept (one argument per parameter).
    :param ParamSelector param_selector: what decides what parameters to be
        sent. Note that this is structured so it can utilize partial sweeps
        to inform further search, although the best performance requires
        that it can give more than one trial at a time.
    :param int sweep_cores: the number of cores that are used for sweeping
        parameters. This is important both for being able to replicate the
        results of the sweep and in terms of performance. If this number
        exceeds the number of physical cores, it will be simulated in a way
        that correctly utilizes the available resources.

        A higher number of sweep cores gives more accurate and consistent
        hyperparameters, which is essential for meaningful comparisons
        between architectures. 4 is a reasonable starting place.
    :param HyperparameterSettings hparams: either the hyperparameter settings
        to use or a name of a preset ('fastest', 'fast', 'slow', 'slowest')
        that is used for sweeping hyperparameters. This value will be modified
        to ensure we are simulating sweep_cores correctly if necessary.
    :param str folder: where to save the output to. the main output that
        one will typically want to use is

        folder/points/
            funcres.pt: contains a list of tuples where each tuple is
                of the form (params, lr_min, lr_max, bs, perf_train,
                loss_train, perf_val, loss_val). The performance and
                loss are expressed as an array with one element per
                trial.
            params_lookup.pt: contains a tuple of two dictionaries,
                where the first is params_to_ids and goes from a
                tuple of parameters to the corresponding name of
                the folder in folder/points which is the result
                of the model_manager train, and the second has
                the keys/values swapped (ids to params).

    This returns the unpickled content in folder/points/funcres.pt, which is
    a list of tuples of the form

    .. code:: python

        (params: tuple,
        lr_min: float,
        lr_max: float,
        bs: int,
        perf_train: np.ndarray #  (shape=(trials,)),
        loss_train: np.ndarray #  (shape=(trials,)),
        perf_val: np.ndarray #  (shape=(trials,)),
        loss_val: np.ndarray #  (shape=(trials,))
        )

    As is typical, loss is non-negative and lower is better. Performance is
    between 0 and 1 and higher is better.
    """
    module_name = mutils.fix_imports((module_name,))[0]
    n_physical_cores = psutil.cpu_count(False)
    hparams = hyperparams.get_settings(hparams)

    if sweep_cores > n_physical_cores:
        hparams.lr_min_inits = max(hparams.lr_min_inits, sweep_cores)
        hparams.batch_rn_min_inits = max(
            hparams.batch_rn_min_inits, sweep_cores)
        hparams.batch_pt_min_inits = max(
            hparams.batch_pt_min_inits,
            (sweep_cores // hparams.batch_pts)
        )
        sweep_cores = n_physical_cores

    tasks = ParamsTaskQueue(
        n_physical_cores, sweep_cores, module_name, hparams, folder,
        [param_selector]
    )

    sweeper = Sweeper(module_name, param_selector, tasks, sweep_cores,
                      folder, hparams)
    sweeper.add_tasks()

    disp.dispatch(tasks, n_physical_cores,
                  ('ignite_simple.model_manager',),
                  sweeper.add_tasks)
    sweeper.store_params_lookup()
    sweeper.collate_files()

    with open(os.path.join(folder, 'points', 'funcres.pt'), 'rb') as infile:
        res = pickle.load(infile)

    return res
