"""This module is responsible for finding a good range of values within which
a measure is increasing most rapidly. This works best when the y-values have
already been smoothed."""

import typing
import numpy as np
import scipy.signal

def smooth_window_size(npts: int) -> int:
    """Returns the a heuristic for the window size for smoothing an array with
    the given number of points. Specifically, this is 1/10 the number of points
    or 101, whichever is smaller.

    :param int npts: the number of data points you have

    :returns: suggested window size for smoothing the data
    """
    result = int(npts) // 10
    if result >= 100:
        return 101
    if result < 3:
        return 3
    if result % 2 == 0:
        result += 1
    return result

def autosmooth(arr: np.ndarray, axis: int = -1):
    """Causes scipy.signal.savgol_filter on the given data for the given
    dimension with the default settings.

    :param arr: the array to smooth
    :param dim: the smooth dimension
    """
    window_size = smooth_window_size(arr.shape[axis])
    return scipy.signal.savgol_filter(arr, window_size, 1, axis=axis)

def nonzero_intervals(vec: np.ndarray) -> np.ndarray:
    """Find which intervals in the given vector are non-zero.

    Adapted from https://stackoverflow.com/a/27642744

    :param np.ndarray vec: the vector to scan for non-zero intervals

    :returns: a list [x1, x2, ..., xn] such that [xi, xi+1) is a nonzero
        interval in vec
    """
    if not vec.shape:
        return []

    edges, = np.nonzero(np.diff((vec == 0).astype('int32')))
    edge_vec = [edges + 1]
    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])

    return np.concatenate(edge_vec)

def trim_range_derivs(xs: np.ndarray,
                      derivs: np.ndarray,
                      x_st_ind: int,
                      x_en_ind: int) -> typing.Tuple[int, int]:
    """Given that we have found a good interval in the xs for which the derivs
    is positive, it may be possible that the bulk of the integral can be
    maintained while reducing the width of the integral. This function returns
    a new interval which is a subset of the given interval for which the
    bulk of the integral is maintained.

    :param np.ndarray xs: with shape `(points,)`, the x-values corresponding
        to the derivatives

    :param np.ndarray derivs: with shape `(points,)`, the relevant derivatives

    :param int x_st_ind: the index in xs the good interval starts at

    :param int x_en_ind: the index in xs the good interval ends at

    :returns: `(x_st_ind, x_end_ind)` which is a subset of the passed interval
    """

    xs_in_range = xs[x_st_ind:x_en_ind]
    in_range = derivs[x_st_ind:x_en_ind]
    max_in_range = in_range.max()

    intvl_width = xs_in_range[-1] - xs_in_range[0]
    intvl_intgrl = np.trapz(in_range, xs_in_range)

    for new_floor_perc in np.linspace(0.5, 0, num=50, endpoint=False):
        new_floor = max_in_range * new_floor_perc
        new_valid = in_range > new_floor
        new_valid_rev = new_valid[::-1]

        first_true = np.argmax(new_valid)
        last_true = new_valid.shape[0] - np.argmax(new_valid_rev)
        # now first_true:last_true is the correct range

        new_intvl_xs = xs_in_range[first_true:last_true]
        new_intvl_derivs = in_range[first_true:last_true]

        new_intvl_width = new_intvl_xs[-1] - new_intvl_xs[0]
        new_intvl_perc = new_intvl_width / intvl_width

        if new_intvl_perc > (1 - new_floor_perc * 1.2):
            # e.g. we shaved off 10% of max and only lost 5% of width,
            # not meaningfully helpful for reducing interval size
            continue

        new_intvl_intgrl = np.trapz(new_intvl_derivs, new_intvl_xs)
        new_intvl_intgrl_perc = new_intvl_intgrl / intvl_intgrl

        if new_intvl_intgrl_perc < (1 - new_floor_perc * 1.2):
            # e.g. we shaved off 10% of max and lost 20% of integral,
            # not meaningfully helpful for reducing interval size
            continue

        if new_intvl_intgrl_perc < (1 - new_intvl_perc * 1.2):
            # e.g. we shaved off 15% of width and lost 20% of integral,
            # not meaningfully helpful for reducing interval size
            continue

        # now e.g. shaved off 10% of max, lost 15% width but only
        # 5% of the integral
        return (x_st_ind + first_true, x_st_ind + last_true)

    return (x_st_ind, x_en_ind)

def find_with_derivs(xs: np.ndarray,
                     derivs: np.ndarray,
                     select_with_width_only: bool = False) -> typing.Tuple[float, float]:
    """Finds the range in derivs wherein the derivative is always
    positive. From these intervals, this returns specifically the one with
    the greatest integral. If select_with_width_only is true, we find the
    interval with the greatest width amongst those found instead.

    :param np.ndarray xs: with shape `(points,)`, the x-values corresponding to
        the derivatives

    :param np.ndarray derivs: with shape `(points,)`, the relevant derivatives

    :param bool select_with_width_only: Change the reason for selecting
        intervals from greatest integral to greatest width in xs. This
        is helpful if least sensitivity is preferred or one is concerned
        that the derivs are very noisy

    :returns: the `(min, max)` for the best interval in xs that has positive
        derivative in ys. Where multiple such intervals exist, this is the
        one with the greatest integral
    """
    derivs = derivs.copy()
    derivs[derivs < 0] = 0
    candidates = nonzero_intervals(derivs)

    if not candidates.shape or candidates.shape[0] < 2:
        return xs[0], xs[-1]

    best_change = 0
    best_candidate = -1
    for i in range(candidates.shape[0] - 1):
        st = candidates[i]
        en = candidates[i + 1]
        change = (abs(xs[min(en, len(derivs) - 1)] - xs[st])
                  if select_with_width_only
                  else np.trapz(derivs[st:en], xs[st:en]))
        if change > best_change:
            best_candidate = i
            best_change = change

    x_st_ind = candidates[best_candidate]
    x_en_ind = candidates[best_candidate + 1]

    x_st_ind, x_en_ind = trim_range_derivs(xs, derivs, x_st_ind, x_en_ind)

    return (xs[x_st_ind], xs[x_en_ind - 1])
