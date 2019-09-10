"""Contains the interface and some implementations for how parameters are
selected for a particular point. Method selection techniques which do not
depend on the results of previous points are more easily parallelizable.
"""
import numpy as np
import random
import typing

class SweepListener:
    """An interface for something which is listening for the result of a
    sweep.
    """
    def on_hparams_found(self, values: tuple, lr_min: float, lr_max: float,
                         batch_size: int) -> None:
        """Called when the parameter sweep has found the appropriate hyper
        parameters for the given set of main parameters.

        :param tuple values: each element corresponds to the value of the
            corresponding parameter. For example, if the parameters list
            is actuallly "nonlinearity", "hidden size" then values might
            be "Tanh", 128

        :param float lr_min: the best value for the minimum for the learning
            rate during one oscillation of learning rate
        :param float lr_max: the best value for the maximum for the learning
            rate during one oscillation of learning rate
        :param int batch_size: the best batch size to use
        """
        pass

    def on_trials_completed(self, values: tuple, perfs_train: np.ndarray,
                            losses_train: np.ndarray,
                            perfs_val: np.ndarray,
                            losses_val: np.ndarray):
        """Invoked when we perform some trials. This could be called multiple
        times of the same point at the decision of the sweeper, but each
        trial will only be sent to this function once.

        :param tuple values: the values for each parameter in order
        :param np.ndarray perfs_train: the final performance for each trial
            performed, so has shape (trials,), on the training dataset
        :param np.ndarray losses_train: the final loss for each trial
            performed, so has shape (trials,), on the training dataset
        :param np.ndarray perfs_val: the final performance for each trial
            performed, so has shape (trials,), on the validation dataset
        :param np.ndarray losses_val: the final loss for each trial
            performed, so has shape (trials,), on the validation dataset
        """
        pass

class ParamSelector(SweepListener):
    """The interface for a class which is capable of selecting what values
    for a given known set of parameters should have models initialized
    and compared.
    """

    def get_next_trials(self) -> typing.Optional[tuple]:
        """Get the next set of trials to perform. The result will be a tuple
        of two elements - the values (a tuple) and the number of trials to
        perform. This may return the same values multiple times, although
        it may improve performance to batch trials of the same values
        together.

        This may return None only if the ParamSelector is waiting on the result
        of previous trials before it will continue. After returning None, this
        function will be invoked after the next on_trials_completed or
        on_hparams_found call. It may return None multiple times in a row if
        there were multiple function calls which it's waiting for.

        Note it should return None, not (None, None)

        :return: A tuple of 2 elements, the first being a tuple of the values
            for the parameters and the second being a nonnegative int for the
            number of trials to perform. If the int is 0, the hparams will be
            found if they haven't already but no trials will be performed.
            If the result is None, see above.
        :rtype: typing.Optional[tuple]
        """
        raise NotImplementedError

    def has_more_trials(self) -> bool:
        """Determines if this param selector will ever want to perform more
        trials. If False, the sweeper will stop calling get_next_trials and
        this must return False at least until the sweep completes. If
        get_next_trials should continue to be called this should return True.
        """
        raise NotImplementedError

class FixedSweep(ParamSelector):
    """A sweep which has a fixed order of results from get_next_trials.

    :ivar tuple gnt_results: the results for get_next_trials in the order
        they should appear
    :ivar int gnt_results_index: the index in gnt_results for the next
        invocation of get_next_trials.
    """
    def __init__(self, gnt_results: tuple) -> None:
        self.gnt_results = gnt_results
        self.gnt_results_index = 0

    def get_next_trials(self):
        self.gnt_results_index += 1
        return self.gnt_results[self.gnt_results_index - 1]

    def has_more_trials(self):
        return self.gnt_results_index < len(self.gnt_results)

    @classmethod
    def with_fixed_trials(cls, pts: typing.Iterable[tuple],
                          trials: int) -> 'FixedSweep':
        """For each of the given points, perform precisely the specified
        number of trials.

        :param pts: the points to test
        :type pts: iterable[tuple]
        :param trials: the numebr of trials to perform at each point
        :type trials: int
        :returns: the corresponding parameter selector
        :rtype: FixedSweep
        """
        return cls(tuple((pt, trials) for pt in pts))

    @classmethod
    def with_random_trials(
            cls, ranges: typing.Tuple[
                typing.Union[int, set, tuple]],
            points: int, trials: int, r_eps: float = 0.01) -> 'FixedSweep':
        """For each parameter, ranges contains the valid values for that
        parameter. If the range is an int, it is assumed to be the arange
        for that int + 1. If it is a set, it contains the valid values for
        that parameter. If it's a tuple, it should contain two elements, the
        min and the max, where both are integers if the values can only be
        integers and either is a float if the values can be floats.

        Ranges are inclusive.

        :param ranges: the possible values for each parameter, one set of
            possible values per parameter.
        :type ranges: typing.Union[int, set, tuple]
        :param points: the number of unique points to select
        :type points: int
        :param trials: the number of trials per point
        :type trials: int
        :param r_eps: Only used if one of the ranges is specified as an
            interval (i.e., not a set). Points which would be within
            (r_eps*interval_width) of each other are rejected during sampling.
            Should be less than 1/(2 * trials) for the randomness to be
            meaningfully different from grid. If too large an error may be
            thrown as not enough trials could be found that weren't
            rejected.
        """

        # convert ints into the corresponding integer range, lists into tuples
        # for convenience (since lists usually can be substituted in where we
        # expect tuples), and then convert sets to lists. We will use the list
        # type to indicate treating it like a set instead of the set type, b/c
        # random.choice expects list-like not set-like.

        # We also ensure that if any element is a float than so is the first
        # one for ranges.

        ranges = list(ranges)
        for i in range(len(ranges)):
            if isinstance(ranges[i], int):
                ranges[i] = (0, ranges[i])
            elif isinstance(ranges[i], list):
                ranges[i] = tuple(ranges[i])
            elif isinstance(ranges[i], (set, frozenset)):
                ranges[i] = list(ranges[i])

            if isinstance(ranges[i], tuple):
                if isinstance(ranges[i][1], float):
                    ranges[i] = (float(ranges[i][0]), ranges[i][1])



        results = []
        def should_reject(pms):
            for res in results:
                num_same = 0
                for (rge, v1, v2) in zip(ranges, res, pms):
                    are_same = v1 == v2
                    if (v1 == v2 or (
                            isinstance(rge, tuple)
                            and abs(v1 - v2) <= r_eps * (rge[1] - rge[0]))):
                        num_same += 1
                if num_same == len(pms):
                    return True
            return False

        n_rejections = 0
        while len(results) < points:
            params = []
            for rg in ranges:
                if isinstance(rg, list):
                    params.append(random.choice(rg))
                elif isinstance(rg[0], float):
                    params.append(random.uniform(rg[0], rg[1]))
                else:
                    params.append(random.randint(rg[0], rg[1]))
            if not should_reject(params):
                results.append(tuple(params))
            else:
                n_rejections += 1
                if n_rejections > points * 2:
                    raise ValueError('exceeded max number of rejections: '
                                     + str(n_rejections))

        return cls.with_fixed_trials(results, trials)

