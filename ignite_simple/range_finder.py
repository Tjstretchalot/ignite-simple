"""This module is responsible for finding a good range of values within which
a measure is increasing most rapidly. This works best when the y-values have
already been smoothed."""

import typing
import numpy as np

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
    if result % 2 == 0:
        result += 1
    return result

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
    edge_vec = [edges+1]
    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])

    return np.concatenate(edge_vec)

def find_with_derivs(xs: np.ndarray, derivs: np.ndarray
                    ) -> typing.Tuple[int, int]:
    """Finds the range in derivs wherein the derivative is always
    positive. From these intervals, this returns specifically the one with
    the greatest integral.

    :param np.ndarray xs: with shape `(points,)`, the x-values corresponding to
        the derivatives

    :param np.ndarray derivs: with shape `(points,)`, the relevant derivatives

    :returns: the `(start_ind, end_ind)` for the best interval in derivs such
        that the desired points are `pts[start_ind:end_ind]`
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
        change = np.trapz(derivs[st:en], xs[st:en])
        if change > best_change:
            best_candidate = i
            best_change = change

    return (xs[candidates[best_candidate]],
            xs[candidates[best_candidate + 1]])

def find(xs: np.ndarray, ys: np.ndarray) -> typing.Tuple[float, float]:
    """Finds the range (min, max) of xs over which the derivative of y is
    positive and during which we saw the greatest change in x. This requires
    that ys is fairly smooth to produce reasonable results.

    .. note::

        If ys was smoothed, then this will be biased to producing intervals
        right of the true best interval. If the derivative is smoothed instead,
        then find_with_derivs will produce an unbiased interval.

    :param np.ndarray xs: The x-values, with shape (num_pts,)
    :param np.ndarray ys: The y-values, with the same shape as the xs

    :returns: (min, max) of xs where the ys increase the quickest
    """
    deriv = np.gradient(ys, xs)
    deriv[deriv < 0] = 0
    candidates = nonzero_intervals(deriv)

    if not candidates.shape:
        return xs[0], xs[-1]

    best_change = 0
    best_candidate = -1
    for i in range(candidates.shape[0] - 1):
        st = candidates[i]
        en = candidates[i + 1] - 1
        change = ys[en] - ys[st]
        if change > best_change:
            best_candidate = i
            best_change = change

    return (xs[candidates[best_candidate]],
            xs[candidates[best_candidate + 1]])
