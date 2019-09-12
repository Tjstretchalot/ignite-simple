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
from sortedcontainers import SortedList
import logging.config
import os

class Task:
    """A description of a task.

    :ivar str module: the name of the module which contains the callable
    :ivar str attrname: the name of the attribute within the module
    :ivar tuple args: the arguments to pass to the callable
    :ivar dict kwargs: the keyword arguments to pass to the callable
    :ivar optional[int] cores: the number of cores this task will use, None
        for all cores
    :ivar optional[callable] callback: an optional callback for when this task
        completes, not sent over. The callback is invoked on the main thread
    """

    def __init__(self, module: str, attrname: str, args: tuple,
                 kwargs: dict, cores: typing.Optional[int],
                 callback: typing.Optional[typing.Callable] = None):
        self.module = module
        self.attrname = attrname
        self.args = args
        self.kwargs = kwargs
        self.cores = cores
        self.callback = callback

    def worker_version(self) -> 'Task':
        if self.callback is None:
            return self
        return Task(self.module, self.attrname, self.args, self.kwargs,
                    self.cores)

    def __str__(self):
        return f'{self.module}.{self.attrname}(*{self.args}, **{self.kwargs}) [cores={self.cores}]'

class TaskQueue:
    """An interface for something which is capable of returning the next
    task to perform based on the number of available cores.
    """

    def set_total_cores(self, total_cores: int) -> None:
        """Clips the maximum number of cores that each task requires to not
        exceed total_cores.
        """
        raise NotImplementedError

    def get_next_task(self, cores: int) -> typing.Optional[Task]:
        """Gets the next task to perform given that there are the specified
        number of cores available. The resulting Task should require fewer
        than the number of cores available. May return None if there are
        no tasks which meet the requirements to do.

        :param cores: the number of cores available
        :type cores: int
        :return: the task to start work on, None if no task available yet
        :rtype: typing.Optional[Task]
        """
        raise NotImplementedError

    def have_more_tasks(self) -> bool:
        """Returns True if there is at least one more task to do, and returns
        False if there are no more tasks to do
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """An estimate for how many tasks remain in this queue."""
        return float('inf')

class GreedyTaskQueue(TaskQueue):
    """Describes a greedy task queue, which prioritizes tasks with the highest
    number of cores.

    :ivar int total_cores: if the cores on a task is None, we assume it is
        going to require this many cores.
    :ivar dict[int, list[Task]] tasks_by_core: a lookup for remaining tasks,
        where the key is the number of cores the task requires.
    :ivar SortedList[int] sorted_keys: the keys for tasks_by_core in
        ascending order.
    :ivar bool expecting_more_tasks: if True, we expect that add_task will be
        called in the (near) future, hence we always return True from
        have_more_tasks
    :ivar int _len: the length of this queue
    """
    def __init__(self, total_cores: int):
        """Initializes an empty task queue."""
        self.total_cores = total_cores
        self.tasks_by_core = dict()
        self.sorted_keys = SortedList()
        self.expecting_more_tasks = False
        self._len = 0

    def flatten(self) -> typing.List[Task]:
        """Returns this task queue as a list of tasks.
        """
        res = []
        for v in self.tasks_by_core.values():
            res.extend(v)
        return res

    def set_total_cores(self, total_cores):
        old_total = self.total_cores
        self.total_cores = total_cores
        if len(self.sorted_keys) == 0:
            return
        if old_total < total_cores:
            flat = self.flatten()
            self.sorted_keys = SortedList()
            self.tasks_by_core = dict()
            self._len = 0
            self.add_tasks(flat)
            return

        arrs = []
        while self.sorted_keys[-1] > self.total_cores:
            k = self.sorted_keys.pop()
            arrs.append(self.tasks_by_core[k])
            del self.tasks_by_core[k]

        if not self.sorted_keys:
            self.sorted_keys.append(self.total_cores)
            self.tasks_by_core[self.total_cores] = []

        last = self.tasks_by_core[self.sorted_keys[-1]]
        for arr in arrs:
            last.extend(arr)


    def add_task(self, task: Task) -> None:
        """Adds the given task to this queue."""
        cores_req = self.total_cores
        if task.cores is not None:
            cores_req = min(task.cores, self.total_cores)

        arr = self.tasks_by_core.get(cores_req)
        if arr is None:
            arr = []
            self.tasks_by_core[cores_req] = arr
            self.sorted_keys.add(cores_req)
        arr.append(task)
        self._len += 1

    def add_tasks(self, tasks: typing.Iterable[Task]) -> None:
        """Adds all the specified tasks to this queue."""
        for task in tasks:
            self.add_task(task)

    def get_next_task(self, cores: int):
        prev = None
        for k in self.sorted_keys:
            if k > cores:
                break
            prev = k

        if prev is None:
            return None

        arr = self.tasks_by_core[prev]
        res = arr.pop()
        self._len -= 1
        if not arr:
            del self.tasks_by_core[prev]
            self.sorted_keys.remove(prev)
        return res

    def have_more_tasks(self):
        return not self.expecting_more_tasks and len(self.sorted_keys) > 0

    def __len__(self):
        return self._len

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
    :ivar optional[callable] callback: a callback function which will be
        invoked when the current task completes, or None if there is no
        current task or no callback registered
    :ivar float sleep_delay: the current sleep delay for this worker
    """
    def __init__(self, proc: mp.Process, jobq: ZeroMQQueue, ackq: ZeroMQQueue,
                 sleep_delay: float):
        self.process = proc
        self.jobq = jobq
        self.ackq = ackq
        self.cores = None
        self.callback = None
        self.sleep_delay = sleep_delay

    def send(self, task: Task):
        """Tells the worker to perform the given task.

        :param task: the task to perform
        """
        if self.cores:
            raise ValueError('still working!')

        self.jobq.put(('task', task.worker_version()))
        self.cores = task.cores
        self.callback = task.callback

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
            if self.callback is not None:
                self.callback()
                self.callback = None
            return True
        except Empty:
            return False

    def close(self):
        """Shuts down the worker"""
        self.jobq.put(('shutdown',))
        self.process.join()
        self.is_ready() # try to ensure callbacks handled
        self.jobq.close()
        self.ackq.close()
        self.process = None
        self.jobq = None
        self.ackq = None


def _dispatcher(imps, jobq, ackq, sleep_delay):
    for imp in imps:
        importlib.import_module(imp)

    if os.path.exists('logging-worker.conf'):
        logging.config.fileConfig('logging-worker.conf')

    logger = logging.getLogger(__name__)
    logger.debug('Worker initialized')
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
            logger.exception('Dispatcher worker encountered an error')
        ackq.put('ack')

    jobq.close()
    ackq.close()

def dispatch(tasks: typing.Union[TaskQueue, typing.Iterable[Task]],
             total_cores: int,
             suggested_imports: typing.Tuple[str] = tuple(),
             wait_function: typing.Callable = None) -> None:
    """Dispatches the given tasks using greedy selection such that no more
    than the specified number of cores are in use at once, where possible
    to do so. This uses a greedy selection of tasks.

    :param tasks: an iterable of tasks to dispatch, or a task queue. May be
        modified, but the logging may be incorrect if modified outside of
        the wait function.
    :param total_cores: the target number of cores to use at once
    :param suggested_imports: things which are imported in each worker process
        during the spawning phase, which causes jobs to be processed more
        smoothly.
    :param wait_function: if not None, this will be invoked when the dispatcher
        is idling. Expected to be fairly fast, and it is safe to modify tasks
        within this call.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(tasks, TaskQueue):
        que: TaskQueue = GreedyTaskQueue(total_cores)
        que.add_tasks(tasks)
        tasks = que
        del que

    ntasks = len(tasks)

    if total_cores <= 1:
        last_print = time.time()

        skips = 0
        i = 0
        while tasks.have_more_tasks():
            old_len = len(tasks)
            task = tasks.get_next_task(1)
            ntasks += len(tasks) - old_len
            if task is None:
                skips += 1
                if skips > 1000:
                    logger.warn(
                        'TaskQueue have_more_tasks is True but '
                        + 'get_next_task is has been None for a long time')
                    skips = 0

                if wait_function is not None:
                    old_len = len(tasks)
                    wait_function()
                    ntasks += len(tasks) - old_len
                time.sleep(0.1)
                continue
            ntasks += 1
            skips = 0
            i += 1

            mod = importlib.import_module(task.module)
            tocall = getattr(mod, task.attrname)
            tocall(*task.args, **task.kwargs)
            if time.time() - last_print > 5:
                logger.debug(f'Finished task {i}/{ntasks}...')
        return

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
    while tasks.have_more_tasks():
        cores_in_use = 0
        for proc in processes:
            if not proc.is_ready():
                cores_in_use += proc.cores

        avail_cores = total_cores - cores_in_use
        old_len = len(tasks)
        task = tasks.get_next_task(avail_cores)
        ntasks += len(tasks) - old_len
        if task is None:
            for proc in processes:
                if (not proc.cores) and (proc.sleep_delay > 0.02):
                    proc.update_sleep_delay(0.1)

            if (tasks_dispatched > last_printed_tasks
                    and time.time() - last_printed_at > 5):
                logger.debug('Dispatched %s/%s tasks... (currently using %s/%s'
                             + ' cores)', tasks_dispatched, ntasks,
                             cores_in_use, total_cores)
                last_printed_tasks = tasks_dispatched
                last_printed_at = time.time()
            if wait_function is not None:
                old_len = len(tasks)
                wait_function()
                ntasks += len(tasks) - old_len
            time.sleep(0.1)
            continue

        ntasks += 1

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
        cores_in_use += task.cores or total_cores

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
