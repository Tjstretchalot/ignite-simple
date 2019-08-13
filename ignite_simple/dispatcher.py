"""This package handles accepting a list of jobs and then deciding which of
them to run in what order.

This module is intended for use by the analysis module for managing all the
jobs it has to do. The analysis package tends to have many jobs are all fairly
independent of each other. Things like rq and celery are way over-engineered
for this task.
"""
import typing
import importlib
import time
import multiprocessing as mp
import logging
from queue import Empty
from pympanim.zeromqqueue import ZeroMQQueue
import traceback

class Task:
    """A description of a task.

    :ivar str module: the name of the module which contains the callable
    :ivar str attrname: the name of the attribute within the module
    :ivar tuple args: the arguments to pass to the callable
    :ivar dict kwargs: the keyword arguments to pass to the callable
    :ivar optional[int] cores: the number of cores this task will use, None
        for all cores
    """

    def __init__(self, module: str, attrname: str, args: tuple,
                 kwargs: dict, cores: typing.Optional[int]):
        self.module = module
        self.attrname = attrname
        self.args = args
        self.kwargs = kwargs
        self.cores = cores

class MainToWorkerConnection:
    """Describes a connection from the main process to the worker process

    :ivar Process process: the actual worker process
    :ivar ZeroMQQueue jobq: the queue that jobs can be sent through to this
        worker
    :ivar ZeroMQQueue ackq: the queue that the worker sends job completed
        notifications through
    :ivar optional[int] cores: the number of cores that this worker is
        currently using, or None if the worker is not doing anything right
        now
    :ivar float sleep_delay: the current sleep delay for this worker
    """
    def __init__(self, proc: mp.Process, jobq: ZeroMQQueue, ackq: ZeroMQQueue,
                 sleep_delay: float):
        self.process = proc
        self.jobq = jobq
        self.ackq = ackq
        self.cores = None
        self.sleep_delay = sleep_delay

    def send(self, task: Task):
        """Tells the worker to perform the given task.

        :param task: the task to perform
        """
        if self.cores:
            raise ValueError('still working!')

        self.jobq.put(('task', task))
        self.cores = task.cores

    def update_sleep_delay(self, delay: float):
        """Tells the worker to poll for jobs with the given inter-poll delay"""
        self.jobq.put(('change_sleep_delay', delay))
        self.sleep_delay = delay

    def is_ready(self) -> bool:
        """Checks to see if the worker is ready to get a new task.
        """
        if not self.cores:
            return True

        try:
            self.ackq.get_nowait()
            self.cores = None
            return True
        except Empty:
            return False

    def close(self):
        """Shuts down the worker"""
        self.jobq.put(('shutdown',))
        self.process.join()
        self.jobq.close()
        self.ackq.close()
        self.process = None
        self.jobq = None
        self.ackq = None


def _dispatcher(imps, jobq, ackq, sleep_delay):
    for imp in imps:
        importlib.import_module(imp)

    jobq = ZeroMQQueue.deser(jobq)
    ackq = ZeroMQQueue.deser(ackq)

    while True:
        try:
            job = jobq.get_nowait()
        except Empty:
            time.sleep(sleep_delay)
            continue

        if job[0] == 'shutdown':
            break
        if job[0] == 'change_sleep_delay':
            sleep_delay = job[1]
            continue

        task = job[1]
        try:
            mod = importlib.import_module(task.module)
            tocall = getattr(mod, task.attrname)
            tocall(*task.args, **task.kwargs)
        except:  # noqa: E722
            traceback.print_exc()
        ackq.put('ack')

    jobq.close()
    ackq.close()

def dispatch(tasks: typing.Tuple[Task], total_cores: int,
             suggested_imports: typing.Tuple[str] = tuple()) -> None:
    """Dispatches the given tasks using greedy selection such that no more
    than the specified number of cores are in use at once, where possible
    to do so. This uses a greedy selection of tasks.

    :param tasks: an iterable of tasks to dispatch
    :param total_cores: the target number of cores to use at once
    :param suggested_imports: things which are imported in each worker process
        during the spawning phase, which causes jobs to be processed more
        smoothly.
    """
    logger = logging.getLogger(__name__)

    if total_cores <= 1:
        last_print = time.time()
        for i, task in enumerate(tasks):
            mod = importlib.import_module(task.module)
            tocall = getattr(mod, task.attrname)
            tocall(*task.args, **task.kwargs)
            if time.time() - last_print > 5:
                print(f'Finished task {i+1}/{len(tasks)}...')
        return

    tasks_by_cores = dict()
    smallest_num_cores = total_cores
    for task in tasks:
        cores = task.cores
        if cores is None:
            cores = total_cores
        elif cores > total_cores:
            cores = total_cores
        elif cores < 1:
            cores = 1
        task.cores = cores

        if cores not in tasks_by_cores:
            tasks_by_cores[cores] = [task]
        else:
            tasks_by_cores[cores].append(task)

        smallest_num_cores = min(smallest_num_cores, cores)

    processes: typing.List[MainToWorkerConnection] = []

    for _ in range(total_cores):
        jobq = ZeroMQQueue.create_send()
        ackq = ZeroMQQueue.create_recieve()
        proc = mp.Process(
            target=_dispatcher,
            args=(suggested_imports, jobq.serd(), ackq.serd(), 0.01)
        )
        proc.start()
        processes.append(MainToWorkerConnection(proc, jobq, ackq, 0.01))

    tasks_dispatched = 0
    last_printed_at = time.time()
    last_printed_tasks = 0
    while tasks_by_cores:
        cores_in_use = 0
        for proc in processes:
            if not proc.is_ready():
                cores_in_use += proc.cores

        avail_cores = total_cores - cores_in_use
        if avail_cores < smallest_num_cores:
            for proc in processes:
                if (not proc.cores) and (proc.sleep_delay > 0.02):
                    proc.update_sleep_delay(0.1)

            if (tasks_dispatched > last_printed_tasks
                    and time.time() - last_printed_at > 5):
                logger.debug('Dispatched %s/%s tasks...',
                             tasks_dispatched, len(tasks))
                last_printed_tasks = tasks_dispatched
                last_printed_at = time.time()
            time.sleep(0.1)
            continue

        cores_to_use = avail_cores
        while cores_to_use not in tasks_by_cores:
            cores_to_use -= 1

        to_choose_from = tasks_by_cores[cores_to_use]
        task = to_choose_from.pop()
        if not to_choose_from:
            del tasks_by_cores[cores_to_use]

            if tasks_by_cores and cores_to_use == smallest_num_cores:
                logger.debug('Completed all tasks that require %s core%s...',
                             smallest_num_cores,
                             's' if smallest_num_cores > 1 else '')
                while smallest_num_cores not in tasks_by_cores:
                    smallest_num_cores += 1

        proc_to_use = None
        for proc in processes:
            if not proc.cores:
                if proc_to_use is None or proc.sleep_delay < proc_to_use.sleep_delay:
                    proc_to_use = proc
                if proc.sleep_delay < 0.02:
                    break

        if proc_to_use.sleep_delay > 0.02:
            proc.update_sleep_delay(0.01)

        proc.send(task)

        tasks_dispatched += 1
        cores_in_use += cores_to_use

    for proc in processes:
        proc: MainToWorkerConnection
        proc.update_sleep_delay(0.1)

    cnt_rem = 0
    for proc in processes:
        if not proc.is_ready():
            cnt_rem += 1

    if cnt_rem > 0:
        logger.debug('Waiting on %s remaining tasks...', cnt_rem)

    for proc in processes:
        proc.is_ready()
        proc.close()
