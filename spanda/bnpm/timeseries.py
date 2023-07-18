import numpy as np
import time
from matplotlib import pyplot as plt
import copy
from numba import njit, jit, prange
import pandas as pd
import torch


def event_triggered_traces(
    arr,
    idx_triggers,
    win_bounds=[-100, 0],
    dim=0,
    verbose=1,
):
    """
    Makes event triggered traces along last dimension.
    New version using torch.
    RH 2022

    Modified a bit so that the output shape is (num.of.idx_triggers, win_bounds, num.of.other dimensions)
    First two dimensions are the event triggered traces.
    GH 2023

    Args:
        arr (np.ndarray or torch.Tensor):
            Input array. Dimension 'dim' will be
             aligned to the idx specified in 'idx_triggers'.
        idx_triggers (list of int, np.ndarray, or torch.Tensor):
            1-D boolean array or 1-D index array.
            True values or idx values are trigger events.
        win_bounds (size 2 integer list, tuple, or np.ndarray):
            2 value integer array. win_bounds[0] should
             be negative and is the number of samples prior
             to the event that the window starts.
             win_bounds[1] is the number of samples
             following the event.
            Events that would have a window extending
             before or after the bounds of the length
             of the trace are discarded.
        dim (int):
            Dimension of 'arr' to align to 'idx_triggers'.
            Sampling windows are taken along this dimension.
        verbose (int):
            0: no print statements
            1: print warnings
            2: print warnings and info

     Returns:
        et_traces (np.ndarray or torch.Tensor):
             Event Triggered Traces. et_traces.ndim = arr.ndim+1.
             First two dims are the event triggered traces.
             Shape: (len(idx_triggers), len(window), lengths of other dimensions besides 'dim')
        xAxis (np.ndarray or torch.Tensor):
            x-axis of the traces. Aligns with dimension
             et_traces.shape[1]
        windows (np.ndarray or torch.Tensor):
            Index array of the windows used.
    """
    from warnings import warn

    ## Error checking
    assert isinstance(dim, int), "dim must be int"
    assert isinstance(
        arr, (np.ndarray, torch.Tensor)
    ), "arr must be np.ndarray or torch.Tensor"
    assert isinstance(
        idx_triggers, (list, np.ndarray, torch.Tensor)
    ), "idx_triggers must be list, np.ndarray, or torch.Tensor"
    assert isinstance(
        win_bounds, (list, tuple, np.ndarray)
    ), "win_bounds must be list, tuple, or np.ndarray"
    ## Warn if idx_triggers are not integers
    if isinstance(idx_triggers, np.ndarray) and not np.issubdtype(
        idx_triggers.dtype, np.integer
    ):
        warn(
            "idx_triggers is np.ndarray but not integer dtype. Converting to torch.long dtype."
        )
    if isinstance(idx_triggers, torch.Tensor) and not torch.is_tensor(
        idx_triggers, dtype=torch.long
    ):
        warn(
            "idx_triggers is torch.Tensor but not dtype torch.long. Converting to torch.long dtype."
        )
    if isinstance(idx_triggers, list):
        warn(
            "Using a list for idx_triggers is slow. Convert to np.array or torch.Tensor."
        )
        if all([isinstance(i, int) for i in idx_triggers]):
            warn(
                "idx_triggers is list but not all elements are integers. Converting to torch.long dtype."
            )

    ## if idx_triggers is length 0, return empty arrays
    if len(idx_triggers) == 0:
        print(
            "idx_triggers is length 0. Returning empty arrays."
        ) if verbose > 0 else None
        arr_out = np.empty(
            (0, win_bounds[1] - win_bounds[0], *arr.shape[:dim], *arr.shape[dim + 1 :])
        )
        xAxis_out = np.empty((0, win_bounds[1] - win_bounds[0]))
        windows_out = np.empty((0, win_bounds[1] - win_bounds[0]))
        return arr_out, xAxis_out, windows_out

    dtype_in = (
        "np"
        if isinstance(arr, np.ndarray)
        else "torch"
        if isinstance(arr, torch.Tensor)
        else "unknown"
    )
    arr = torch.as_tensor(arr)  ## convert to tensor
    xAxis = torch.arange(
        win_bounds[0], win_bounds[1], dtype=torch.long
    )  ## x-axis for each window relative to trigger

    dim = arr.ndim + dim if dim < 0 else dim  ## convert negative dim to positive dim

    isnan = (
        torch.isnan
        if isinstance(idx_triggers, torch.Tensor)
        else np.isnan
        if isinstance(idx_triggers, np.ndarray)
        else None
    )
    idx_triggers_clean = torch.as_tensor(
        idx_triggers[~isnan(idx_triggers)], dtype=torch.long
    )  ## remove nans from idx_triggers and convert to torch.long

    windows = torch.stack(
        [xAxis + i for i in torch.as_tensor(idx_triggers_clean, dtype=torch.long)],
        dim=0,
    )  ## make windows. shape = (n_triggers, len_window)
    win_toInclude = (torch.any(windows < 0, dim=1) == 0) * (
        torch.any(windows > arr.shape[dim], dim=1) == 0
    )  ## boolean array of windows that are within the bounds of the length of 'dim'
    n_win_excluded = torch.sum(
        win_toInclude == False
    )  ## number of windows excluded due to window bounds. Only used for printing currently
    windows = windows[
        win_toInclude
    ]  ## windows after pruning out windows that are out of bounds
    n_windows = windows.shape[0]  ## number of windows. Only used for printing currently

    print(
        f"number of triggers excluded due to window bounds:     {n_win_excluded}"
    ) if (n_win_excluded > 0) and (verbose > 1) else None
    print(
        f"number of triggers included and within window bounds: {len(windows)}"
    ) if verbose > 2 else None

    shape = list(arr.shape)  ## original shape
    dims_perm = (
        [dim] + list(range(dim)) + list(range(dim + 1, len(shape)))
    )  ## new dims for indexing. put dim at dim 0
    arr_perm = arr.permute(*dims_perm)  ## permute to put 'dim' to dim 0
    arr_idx = arr_perm.index_select(
        0, windows.reshape(-1)
    )  ## index out windows along dim 0
    rs = list(arr_perm.shape[1:]) + [
        n_windows,
        win_bounds[1] - win_bounds[0],
    ]  ## new shape for unflattening. 'dim' will be moved to dim -1, then reshaped to n_windows x len_window
    arr_idx_rs = arr_idx.permute(*(list(range(1, arr_idx.ndim)) + [0])).reshape(
        *rs
    )  ## permute to put current 'dim' (currently dim 0) to end, then reshape

    ## GH 2023
    ## Finally, permute back to n_windows x len_window x other_dims
    perm_back = [arr_idx_rs.ndim - 2, arr_idx_rs.ndim - 1] + list(
        range(arr_idx_rs.ndim - 2)
    )
    arr_idx_rs_ordered = arr_idx_rs.permute(*perm_back)

    arr_out = arr_idx_rs_ordered.numpy() if dtype_in == "np" else arr_idx_rs_ordered
    xAxis_out = xAxis.numpy() if dtype_in == "np" else xAxis
    windows_out = windows.numpy() if dtype_in == "np" else windows

    return arr_out, xAxis_out, windows_out
