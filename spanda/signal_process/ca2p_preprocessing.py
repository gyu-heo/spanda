import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import time
import gc

from numba import jit, njit, prange

from pathlib import Path


def make_dFoF(
    F,
    Fneu=None,
    neuropil_fraction=0.7,
    percentile_baseline=30,
    rolling_percentile_window=None,
    roll_centered=True,
    roll_stride=1,
    roll_interpolation="linear",
    channelOffset_correction=0,
    multicore_pref=False,
    verbose=True,
):
    """
    Calculates the dF/F and other signals. Designed for Suite2p data.
    If Fneu is left empty or =None, then no neuropil subtraction done.
    See S2p documentation for more details
    RH 2021-2023

    Args:
        F (np.ndarray):
            raw fluorescence values of each ROI. shape=(ROI#, time)
        Fneu (np.ndarray):
            Neuropil signals corresponding to each ROI. dims match F.
        neuropil_fraction (float):
            value, 0-1, of neuropil signal (Fneu) to be subtracted off of ROI signals (F)
        percentile_baseline (float/int):
            value, 0-100, of percentile to be subtracted off from signals
        rolling_percentile_window (int):
            window size for rolling percentile.
            NOTE: this value will be divided by stride_roll.
            If None, then a single percentile is calculated for the entire trace.
        roll_centered (bool):
            If True, then the rolling percentile is calculated with a centered window.
            If False, then the rolling percentile is calculated with a trailing window
             where the right edge of the window is the current timepoint.
        roll_stride (int):
            Stride for rolling percentile.
            NOTE: rolling_percentile_window will be divided by this value.
            If 1, then the rolling percentile is calculated
             at every timepoint. If 2, then the rolling percentile is calculated at every
             other timepoint, etc.
        roll_interpolation (str):
            Interpolation method for rolling percentile.
            Options: 'linear', 'nearest'
            See pandas.DataFrame.rolling for more details
        channelOffset_correction (float):
            value to be added to F and Fneu to correct for channel offset
        verbose (bool):
            Whether you'd like printed updates
    Returns:
        dFoF (np.ndarray):
            array, dF/F
        dF (np.ndarray):
            array
        F_neuSub (np.ndarray):
            F with neuropil subtracted
        F_baseline (np.ndarray):
            1-D array of size F.shape[0]. Baseline value for each ROI
    """
    from spanda.math_module.timeSeries import percentile_numba

    tic = time.time()

    roll_stride = int(roll_stride)

    F = torch.as_tensor(F, dtype=torch.float32) + channelOffset_correction
    Fneu = (
        torch.as_tensor(Fneu, dtype=torch.float32) + channelOffset_correction
        if Fneu is not None
        else 0
    )

    if Fneu is None:
        F_neuSub = F
    else:
        F_neuSub = F - neuropil_fraction * Fneu

    if rolling_percentile_window is None:
        F_baseline = (
            percentile_numba(F_neuSub.numpy(), ptile=percentile_baseline)
            if multicore_pref
            else np.percentile(F_neuSub.numpy(), percentile_baseline, axis=1)
        )
    else:
        from spanda.math_module.timeSeries import rolling_percentile_pd

        F_baseline = rolling_percentile_pd(
            F_neuSub.numpy()[:, ::roll_stride],
            ptile=percentile_baseline,
            window=int(rolling_percentile_window / roll_stride),
            multiprocessing_pref=multicore_pref,
            prog_bar=verbose,
            center=roll_centered,
            interpolation=roll_interpolation,
        )
    F_baseline = torch.as_tensor(F_baseline, dtype=torch.float32)
    F_baseline = (
        torch.tile(F_baseline[:, :, None], (1, 1, roll_stride)).reshape(
            (F_baseline.shape[0], -1)
        )[:, : F_neuSub.shape[1]]
        if roll_stride > 1
        else F_baseline
    )

    dF = (
        F_neuSub - F_baseline[:, None]
        if F_baseline.ndim == 1
        else F_neuSub - F_baseline
    )
    dFoF = dF / F_baseline[:, None] if F_baseline.ndim == 1 else dF / F_baseline

    if verbose:
        print(
            f"Calculated dFoF. Total elapsed time: {round(time.time() - tic,2)} seconds"
        )

    return dFoF.numpy(), dF.numpy(), F_neuSub.numpy(), F_baseline.numpy()


@njit(parallel=True)
def peter_noise_levels(dFoF, frame_rate):
    """ "
    adapted from:  Peter Rupprecht (github: CASCADE by PTRRupprecht)

    Computes the noise levels for each neuron of the input matrix 'dFoF'.

    The noise level is computed as the median absolute dF/F difference
    between two subsequent time points. This is a outlier-robust measurement
    that converges to the simple standard deviation of the dF/F trace for
    uncorrelated and outlier-free dF/F traces.

    Afterwards, the value is divided by the square root of the frame rate
    in order to make it comparable across recordings with different frame rates.


    input: dFoF (matrix with n_neurons x time_points)
    output: vector of noise levels for all neurons

    """

    def abs_numba(X):
        Y = np.zeros_like(X)
        for ii in prange(len(X)):
            Y[ii] = abs(X[ii])
        return Y

    noise_levels = np.zeros(dFoF.shape[0])
    for ii in prange(dFoF.shape[0]):
        noise_levels[ii] = np.median(abs_numba(np.diff(dFoF[ii, :], 1)))

    # noise_levels = np.median(np.abs(np.diff(dFoF, axis=1)), axis=1)
    noise_levels = (
        noise_levels / np.sqrt(frame_rate) * 100
    )  # scale noise levels to percent
    return noise_levels


def snr_autoregressive(
    x,
    axis=1,
    center=True,
    standardize=True,
    device="cpu",
    return_numpy=True,
    return_cpu=True,
):
    """
    Calculate the SNR of an autoregressive signal.
    Relies on the assumption that the magnitude of the signal
     can be estimated as the correlation of a signal and
     its autoregressive component (corr(sig, roll(sig, 1))).
    RH 2023

    Args:
        x (np.ndarray):
            2D array of shape (n_traces, n_samples)
        axis (int, optional):
            Axis along which to calculate the SNR. Defaults to 1.
        center (bool, optional):
            Whether to center the data before calculating the SNR.
            Defaults to True.
        standardize (bool, optional):
            Whether to standardize the data before calculating the SNR.
              Defaults to True.

    Returns:
        snr (np.ndarray):
            1D array of shape (n_traces,) containing the SNR of each trace.
        s (np.ndarray):
            1D array of shape (n_traces,) containing the signal variance of each trace.
        n (np.ndarray):
            1D array of shape (n_traces,) containing the noise variance of each trace.
    """
    x_norm = torch.as_tensor(x, dtype=torch.float32, device=device)

    if center:
        x_norm = x_norm - torch.mean(x_norm, axis=axis, keepdims=True)
    if standardize:
        x_norm = x_norm / torch.std(x_norm, axis=axis, keepdims=True)

    var = (x_norm**2).sum(axis) / (
        x_norm.shape[axis] - 1
    )  ## total variance of each trace

    s = (x_norm[:, 1:] * x_norm[:, :-1]).sum(axis) / (
        x_norm.shape[axis] - 2
    )  ## signal variance of each trace based on assumption that trace = signal + noise, signal is autoregressive, noise is not autoregressive
    n = var - s
    snr = s / n

    if return_numpy:
        snr, s, n = snr.cpu().numpy(), s.cpu().numpy(), n.cpu().numpy()
    elif return_cpu:
        snr, s, n = snr.cpu(), s.cpu(), n.cpu()

    return snr, s, n


def derivative_MAD(
    X,
    n=(0, 1, 2),
    dt=1,
    axis=1,
    center=True,
    standardize=False,
    device="cpu",
    return_numpy=True,
    return_cpu=True,
):
    """
    Calculate the median absolute deviance of the nth derivatives of a signal.
    This is a generalization of Peter Rupperecht's noise level calculation for
     CASCADE by PTRRupprecht (github.com/PTRRupprecht/CASCADE).
    RH 2023

    Args:
        X (np.ndarray or torch.Tensor):
            Signal to calculate the MAD of the nth derivatives of.
        n (tuple):
            Tuple of integers specifying the nth derivatives to calculate.
        dt (float):
            Time step between samples of the signal. 1/frame_rate.
        axis (int):
            Axis along which to calculate the MAD of the nth derivatives.
        center (bool):
            Whether to center the signal before calculating the MAD.
            Should generally be True
        standardize (bool):
            Whether to standardize the signal before calculating the MAD.
            Should generally be False
        device (str):
            Device to use for torch tensors. 'cpu' or 'cuda'.
        return_numpy (bool):
            Whether to return the results as a numpy array.
        return_cpu (bool):
            Whether to return the results on the cpu.
    """
    n = (n,) if isinstance(n, int) else n

    ## make robust to torch tensor or numpy array inputs of X
    x = torch.as_tensor(X, dtype=torch.float32, device=device)

    x_norm = x - torch.mean(x, axis=axis, keepdim=True) if center else x
    x_norm = x_norm / torch.std(x, axis=axis, keepdim=True) if standardize else x_norm

    ## calculate the nth derivatives of the signal
    x_deriv = [torch.diff(x_norm, n=n_i, dim=axis) for n_i in n]

    ## calculate the median absolute deviation of the nth derivatives
    x_deriv_MAD = [
        torch.median(torch.abs(x_deriv_i), axis=axis)[0] / (dt**-1)
        for x_deriv_i in x_deriv
    ]

    ## return as numpy array if desired
    if return_numpy:
        x_deriv_MAD = [x_deriv_MAD_i.cpu().numpy() for x_deriv_MAD_i in x_deriv_MAD]
    elif return_cpu:
        x_deriv_MAD = [x_deriv_MAD_i.cpu() for x_deriv_MAD_i in x_deriv_MAD]

    return x_deriv_MAD


def trace_quality_metrics(
    F,
    Fneu,
    dFoF,
    F_neuSub,
    F_baseline_roll=None,
    percentile_baseline=30,
    window_rolling_baseline=30 * 60 * 15,
    Fs=30,
    plot_pref=True,
    thresh=None,
    device="cpu",
):
    """
    Some simple quality metrics for calcium imaging traces. Designed to
    work with Suite2p's outputs (F, Fneu) and the make_dFoF function
    above.
    RH 2021

    Args:
        F (np.ndarray or torch.Tensor):
            Fluorescence traces.
            From S2p. shape=[Neurons, Time]
        Fneu (np.ndarray or torch.Tensor):
            Fluorescence Neuropil traces.
            From S2p. shape=[Neurons, Time]
        dFoF (np.ndarray or torch.Tensor):
            Normalized changes in fluorescence ('dF/F').
            From 'make_dFoF' above.
            ((F-Fneu) - F_base) / F_base . Where F_base is
            something like percentile((F-Fneu), 30)
        F_neuSub (np.ndarray or torch.Tensor):
            Neuropil subtracted fluorescence.
            From 'make_dFoF' above
        F_baseline_roll (np.ndarray or torch.Tensor):
            Rolling baseline of F_neuSub.
            From 'make_dFoF' above. If None, then
             will be calculated from F_neuSub.
        percentile_baseline (int, 0 to 100):
            percentile to use as 'baseline'
        window_rolling_baseline (int):
            Window to use for rolling baseline.
            In samples.
        Fs (float):
            Framerate of imaging
        plot_pref (bool):
            Whether to plot the traces and metrics
        thresh:
            Dictionary of thresholds to use.
            Defined as tuples of (min, max) values.
            If None, then use default values:
                'var_ratio__Fneu_over_F': (0, 0.5),
                'EV__F_by_Fneu': (0, 0.5),
                'base_FneuSub': (75, 1500),
                'base_F': (200, 2000),
                'nsr_autoregressive': (0, 6),
                'noise_derivMAD': (0, 0.02),
                'max_dFoF': (0, 10),
                'baseline_var': (0, 0.01),
        device (str):
            Device to use for torch tensors. 'cpu' or 'cuda'.

    Returns:
        tqm: dict with the following fields:
            metrics:
                quality_metrics. Dict of all the
                relevant output variables
            thresh:
                Some hardcoded thresholds for aboslute
                cutoffs
            sign:
                Whether the thresholds for exclusion in tqm_thresh
                should be positive (>) or negative (<). Multiply
                by tqm_thresh to just do everything as (>).
        good_ROIs:
            ROIs that did not meet the exclusion creteria
    """
    from spanda.math_module.similarity import pairwise_orthogonalization_torch

    if F_baseline_roll is None:
        from ..math_module.timeSeries import rolling_percentile_pd

        F_baseline_roll = rolling_percentile_pd(
            F_neuSub.cpu().numpy()[:, ::1]
            if isinstance(F_neuSub, torch.Tensor)
            else F_neuSub[:, ::1],
            ptile=percentile_baseline,
            window=int(window_rolling_baseline),
            multiprocessing_pref=True,
            center=True,
            interpolation="linear",
        )

    F = torch.as_tensor(F, dtype=torch.float32, device=device)
    Fneu = torch.as_tensor(Fneu, dtype=torch.float32, device=device)
    dFoF = torch.as_tensor(dFoF, dtype=torch.float32, device=device)
    F_neuSub = torch.as_tensor(F_neuSub, dtype=torch.float32, device=device)
    F_baseline_roll = torch.as_tensor(
        F_baseline_roll, dtype=torch.float32, device=device
    )

    var_F = torch.var(F, dim=1)
    var_Fneu = torch.var(Fneu, dim=1)

    var_ratio__Fneu_over_F = var_Fneu / var_F
    var_ratio__Fneu_over_F[torch.isinf(var_ratio__Fneu_over_F)] = 0

    # var_FneuSub = torch.var(F_neuSub, dim=1)
    # EV__F_by_Fneu = 1 - (var_FneuSub / var_F)
    _, EV__F_by_Fneu, _, _ = pairwise_orthogonalization_torch(
        v1=F.T,
        v2=Fneu.T,
        center=True,
        device=device,
    )

    # F_baseline = torch.quantile(F, percentile_baseline/100, dim=1, keepdim=True)
    # FneuSub_baseline = torch.quantile(F_neuSub, percentile_baseline/100, dim=1, keepdim=True)
    ## For some reason, torch.quantile is absurdly slow. So we'll use kthvalue instead
    k = (
        int(np.round((F.shape[1] - 1) * percentile_baseline / 100)) + 1
    )  ## kthvalue is 1-indexed
    F_baseline = torch.kthvalue(F, k, dim=1, keepdim=True).values
    FneuSub_baseline = torch.kthvalue(F_neuSub, k, dim=1, keepdim=True).values

    snr_ar, _, _ = snr_autoregressive(
        x=dFoF,
        axis=1,
        center=True,
        standardize=True,
        device=device,
        return_numpy=False,
        return_cpu=False,
    )
    nsr_ar = 1 / snr_ar
    noise_derivMAD = derivative_MAD(
        X=dFoF,
        n=2,
        dt=1 / Fs,
        axis=1,
        center=True,
        standardize=False,
        device=device,
        return_numpy=False,
        return_cpu=False,
    )[0]

    F_baseline_roll_mean = torch.mean(F_baseline_roll, dim=1, keepdim=True)
    dFbrmOverFbrm = (F_baseline_roll - F_baseline_roll_mean) / F_baseline_roll_mean
    baseline_var = torch.var(dFbrmOverFbrm, dim=1)

    max_dFoF = torch.max(dFoF, dim=1).values

    metrics = {
        "var_ratio__Fneu_over_F": var_ratio__Fneu_over_F.cpu().numpy().squeeze(),
        "EV__F_by_Fneu": EV__F_by_Fneu.cpu().numpy().squeeze(),
        "base_FneuSub": FneuSub_baseline.cpu().numpy().squeeze(),
        "base_F": F_baseline.cpu().numpy().squeeze(),
        "nsr_autoregressive": nsr_ar.cpu().numpy().squeeze(),
        "noise_derivMAD": noise_derivMAD.cpu().numpy().squeeze(),
        "max_dFoF": max_dFoF.cpu().numpy().squeeze(),
        "baseline_var": baseline_var.cpu().numpy().squeeze(),
    }
    thresh = (
        {
            "var_ratio__Fneu_over_F": (0, 0.5),
            "EV__F_by_Fneu": (0, 0.7),
            "base_FneuSub": (100, 2000),
            "base_F": (200, 3500),
            "nsr_autoregressive": (0, 6),
            "noise_derivMAD": (0, 0.015),
            "max_dFoF": (0.75, 10),
            "baseline_var": (0, 0.015),
        }
        if thresh is None
        else thresh
    )

    # Exclude ROIs
    good_ROIs = np.ones(dFoF.shape[0], dtype=bool)
    classifications = dict()
    for ii, met in enumerate(metrics):
        to_exclude = (
            (metrics[met] < thresh[met][0])
            + (thresh[met][1] < metrics[met])
            + np.isnan(metrics[met])
        )  # note that if NaN then excluded
        classifications[met] = np.logical_not(to_exclude)
        good_ROIs[to_exclude] = False

    # drop everything into a dict
    tqm = {
        "metrics": metrics,
        "thresh": thresh,
        "classifications": classifications,
    }

    # plot
    if plot_pref:
        fig, axs = plt.subplots(len(tqm["metrics"]), figsize=(7, 10))
        for ii, met in enumerate(tqm["metrics"]):
            axs[ii].hist(
                tqm["metrics"][met][np.where(good_ROIs == 1)[0]], 300, histtype="step"
            )
            axs[ii].hist(
                tqm["metrics"][met][np.where(good_ROIs == 0)[0]], 300, histtype="step"
            )

            axs[ii].title.set_text(
                f"{met}: {np.sum(tqm['classifications'][met]==0)} excl"
            )
            axs[ii].set_yscale("log")

            axs[ii].plot(
                np.array([tqm["thresh"][met], tqm["thresh"][met]]),
                np.array([0, 100]),
                "k",
            )
        fig.legend(("thresh", "included", "excluded"))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.figure()
        plt.plot(good_ROIs)
        plt.plot(scipy.signal.savgol_filter(good_ROIs, 101, 1))

    print(f"ROIs excluded: {int(np.sum(1-good_ROIs))} / {len(good_ROIs)}")
    print(f"ROIs included: {int(np.sum(good_ROIs))} / {len(good_ROIs)}")

    good_ROIs = np.array(good_ROIs, dtype=bool)
    return tqm, good_ROIs
