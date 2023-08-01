import numpy as np
import time
from matplotlib import pyplot as plt
import copy
from numba import njit, jit, prange
import pandas as pd
import torch

from spanda.spanda.util.parallel_helpers import multiprocessing_pool_along_axis


def rolling_percentile_pd(
    X,
    ptile=50,
    window=21,
    interpolation="linear",
    multiprocessing_pref=True,
    prog_bar=False,
    **kwargs_rolling,
):
    """
    Computes a rolling percentile over one dimension
     (defaults to dim 1 / rows).
    This function is currently just a wrapper for pandas'
     rolling library.
    Input can be pandas DataFrame or numpy array, and output
     can also be either.
    I tried to accelerate this with multithreading and numba and
    they don't seem to help or work. Also tried the new
    rolling_quantiles stuff (https://github.com/marmarelis/rolling-quantiles)
    and saw only modest speed ups at the cost of all the
    parameters. I'm still not sure if anyone is using efficient
    insertion sorting.
    RH 2021

    Args:
        X (numpy.ndarray OR pd.core.frame.DataFrame):
            Input array of signals. Calculation done over
            dim 1 (rows).
        ptile (scalar):
            Percentile. 0-100.
        window (scalar):
            Size of window in samples. Ideally odd integer.
        interpolation (string):
            For details see: pandas.core.window.rolling.Rolling.quantile
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.quantile.html
            Can be: linear, lower, higher, nearest, midpoint
        multiprocessing_pref (bool):
            Whether to use multiprocessing to speed up the calculation.
        prog_bar (bool):
            Whether to show a progress bar. For multiprocessing.
        **kwargs_rolling (dict):
            kwargs for pandas.DataFrame.rolling function call.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
            Includes: min_periods, center, axis, win_type, closed

    Returns:
        output (numpy.ndarray OR pd.core.frame.DataFrame):
            Output array of signals.
    """

    if "min_periods" not in kwargs_rolling:
        kwargs_rolling["min_periods"] = 1
    if "center" not in kwargs_rolling:
        kwargs_rolling["center"] = True
    if "axis" not in kwargs_rolling:
        kwargs_rolling["axis"] = 1
    if "win_type" not in kwargs_rolling:
        kwargs_rolling["win_type"] = None
    if "closed" not in kwargs_rolling:
        kwargs_rolling["closed"] = None

    from functools import partial

    _rolling_ptile_pd_helper_partial = partial(
        _rolling_ptile_pd_helper,
        win=int(window),
        ptile=ptile,
        kwargs_rolling=kwargs_rolling,
        interpolation=interpolation,
    )

    if multiprocessing_pref:
        from spanda.spanda.util.parallel_helpers import map_parallel
        from spanda.util.util import make_batches
        import multiprocessing as mp

        n_batches = mp.cpu_count()
        batches = make_batches(X, num_batches=n_batches)
        output = map_parallel(
            _rolling_ptile_pd_helper_partial,
            [list(batches)],
            method="multiprocessing",
            prog_bar=prog_bar,
        )
        output = np.concatenate(output, axis=0)
    else:
        output = _rolling_ptile_pd_helper_partial(X)

    return output


def _rolling_ptile_pd_helper(X, win, ptile, kwargs_rolling, interpolation="linear"):
    return (
        pd.DataFrame(X)
        .rolling(window=win, **kwargs_rolling)
        .quantile(
            ptile / 100,
            numeric_only=True,
            interpolation=interpolation,
            # engine='numba',
            # engine_kwargs={'nopython': True, 'nogil': True, 'parallel': True}
        )
        .to_numpy()
    )


def rolling_percentile_rq(x_in, window, ptile=10, stride=1, center=True):
    import rolling_quantiles as rq

    pipe = rq.Pipeline(
        rq.LowPass(window=window, quantile=(ptile / 100), subsample_rate=stride)
    )
    lag = int(np.floor(pipe.lag))
    if center:
        return pipe.feed(x_in)[lag:]
    else:
        return pipe.feed(x_in)


def rolling_percentile_rq_multicore(
    x_in, window, ptile, stride=1, center=True, n_workers=None
):
    return multiprocessing_pool_along_axis(
        x_in,
        rolling_percentile_rq,
        n_workers=n_workers,
        axis=0,
        **{"window": window, "ptile": ptile, "stride": stride, "center": center},
    )


##############################################################
######### NUMBA implementations of simple algorithms #########
##############################################################
@njit(parallel=True)
def percentile_numba(X, ptile):
    """
    Parallel (multicore) Percentile. Uses numba
    RH 2021

    Args:
        X (ndarray):
            2-D array. Percentile will be calculated
            along second dimension (rows)
        ptile (scalar 0-100):
            Percentile

    Returns:
        X_ptile (ndarray):
            Percentiles of X
    """

    X_ptile = np.zeros(X.shape[0])
    for ii in prange(X.shape[0]):
        X_ptile[ii] = np.percentile(X[ii, :], ptile)
    return X_ptile
