import os
from pathlib import Path
import itertools
import multiprocessing as mp

import numpy as np
import natsort
import torch

import json
import pickle

from tqdm import tqdm

from spanda import _param_defaults_drifter
from spanda import util
from spanda.bnpm.timeseries import event_triggered_traces
from spanda.representation.netrep import LinearMetric


class drifter:
    def __init__(
        self,
        params_input=None,
    ):
        """
        Pipeline class to analyze representational drift.

        All that is neuron does not fire,
        Not all those who DRIFT are lost.

        Args:
            params_input (dict or str, optional):
                Overwrite parameters given inputs. Defaults to None.
                If None, use default parameters.
                If dict, overwrite default parameters' matching key-value pair with given dict.
                If str, load parameters from given json file.
        """
        self.params = _param_defaults_drifter
        self.params_input = params_input

        if params_input is not None:
            self.params = util.overwrite_params(self.params_input, self.params)

    def create_ref_matrices(
        self,
        input_data: dict,
        sampling_events: list,
        sampling_indices: dict,
        sampling_windows: dict,
        sampling_axis: int = 0,
    ) -> list:
        """
        Create reference matrices for each session.
        drifter will compare each sessions' reference matrix to measure the amount of drift.

        Args:
            input_data (dict):
                keys: session index (e.g. date of experiment)
                items: (num.of.samples, num.of.features) shape matrix
            sampling_events (list):
                list of sampling events (e.g. trial_onset, threshold_crossing)
                Used as keys for sampling_indices and sampling_windows
            sampling_indices (dict):
                keys: session index (e.g. date of experiment)
                items: dict of sampling indices for each sampling event
                    keys: sampling event (e.g. trial_onset, threshold_crossing)
                    items: (num.of.sampling_index,) shape array. list, np.ndarray, or torch.Tensor.
            sampling_windows (dict):
                keys: sampling event (e.g. trial_onset, threshold_crossing)
                items: [start, end] of sampling window
            sampling_axis (int, optional):
                Corresponding input_data axis to sample over. Dimension of your num.of.samples. Defaults to 0.
        """
        ref_matrices = []
        for session in input_data:
            full_trace = input_data[session]
            average_trace = []

            ## Iterate over defined sampling events
            for sampling_event in sampling_events:
                sampling_index = sampling_indices[session][sampling_event]
                sampling_window = sampling_windows[sampling_event]

                ## sample_trace: (num.of.sampling_index, sampling_window_size, num.of.features)
                sample_trace, _, _ = event_triggered_traces(
                    arr=full_trace,
                    idx_triggers=sampling_index,
                    win_bounds=sampling_window,
                    dim=sampling_axis,
                )
                ## average_trace: (sampling_window_size, num.of.features)
                average_trace.append(np.nanmean(sample_trace, axis=0))

            ## Concatenate over sampling events to create a single reference matrix
            ## ref_matrices: (sum.of.sampling_window_size, num.of.features)
            ref_matrices.append(np.nan_to_num(np.concatenate(average_trace, axis=0)))
        return ref_matrices

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        weight_method: str = "uniform",
        weights: np.ndarray = None,
    ):
        """
        Find the optimal rotation matrix to align X and Y. Then, return three metrics to quantify the amount of drift.

        Args:
            X (np.ndarray):
                First reference matrix. (num.of.samples, num.of.features)
            Y (np.ndarray):
                Second reference matrix. (num.of.samples, num.of.features)
            weight_method (str, optional):
                Weights angular drift. Defaults to "uniform".
                "uniform": Uniformly weight angular error.
                "EV": Weight angular error by explained variances.
                "custom": Use weights argument to scale angular error.
            weights (np.ndarray, optional):
                Only matters if weight_method is "custom".
                Use weights to scale angular error. Defaults to None.

        Returns:
            self.aligned_dist (float):
                Flattened angular distance between optimally aligned X and Y
                X_aligned is optimally aligned with Y_aligned.
            self.Frobenius_error (float):
                Sum of Frobenius norm between pre- and post-rotation of X and Y (X - X_post, Y - Y_post).
                Measures how big the optimal rotation is.
                Please note that the X_post and Y_post are NOT optimally aligned each other.
                X_post optimally aligns with Y, and vice versa.
            self.angular_dist (float):
                Mean of angular distance between pre- and post-rotation of X and Y.
                If weighted, use weights to scale angular error.
        """
        assert (
            X.shape[1] == Y.shape[1]
        ), "For our standard pipeline, X and Y must have the same number of features"

        metric = LinearMetric(alpha=self.params["netrep_alpha"])
        # metric = LinearMetric(alpha = 1.0)
        metric.fit(X, Y)

        ## Load pre-rotation of X and Y (partially whitened)
        ## If alpha = 1.0, then X_pre = X and Y_pre = Y
        X_pre = X @ metric.Zx_
        Y_pre = Y @ metric.Zy_

        ## Orthogonal Procrustes Problem
        opt_rotation = metric.Rx_ @ metric.Ry_.T

        ## Post-rotation X and Y (partially whitened)
        X_post = X_pre @ opt_rotation
        Y_post = Y_pre @ opt_rotation.T

        ## Angular distance between best-aligned X and Y
        self.aligned_dist = metric.score(X, Y)

        ## Here we use two different metrics to measure the amount of rotation
        ## 1. Frobenius norm of the difference between the pre- and post-rotated matrices
        X_L2_norm = np.linalg.norm(X_pre - X_post, ord="fro")
        Y_L2_norm = np.linalg.norm(Y_pre - Y_post, ord="fro")
        self.Frobenius_error = X_L2_norm + Y_L2_norm

        ## 2. Column-wise angular distance between the pre- and post-rotated matrices
        ## Here, columns are the features (e.g. neurons)
        ## You can weight each features by some values (e.g. variance explained)
        X_col_angles = util.compute_col_angle(
            X_pre, X_post, mean_center=False, verbose=False
        )
        Y_col_angles = util.compute_col_angle(
            Y_pre, Y_post, mean_center=False, verbose=False
        )
        weighted_X_col_angles = util.weight_transform(
            X_col_angles, weights, weight_method
        )
        weighted_Y_col_angles = util.weight_transform(
            Y_col_angles, weights, weight_method
        )

        ## Transform back to the cosine similarity
        angular_error = np.mean((weighted_X_col_angles + weighted_Y_col_angles) / 2)
        self.cossim = np.cos(angular_error)

        return self

    def _pairwise_fit_worker(
        self,
        ii,
        jj,
        X,
        Y,
        weight_method,
        weights,
    ):
        self.fit(X, Y, weight_method, weights)
        return ii, jj, self.aligned_dist, self.Frobenius_error, self.cossim

    def _pairwise_fit_batcher(self, args):
        return self._pairwise_fit_worker(*args)

    def pairwise_fit(
        self,
        ref_matrices: list,
        weight_method: str = "uniform",
        weights: np.ndarray = None,
        processes: int = None,
    ):
        """_summary_

        Args:
            ref_matrices (list):
                List of reference matrices. Each matrix is in a shape of (num.of.samples, num.of.features).
            weight_method (str, optional):
                Weights angular drift. Defaults to "uniform".
                For the details, please refer to the fit function.
            weights (np.ndarray, optional):
                Only matters if weight_method is "custom". Defaults to None.
                For the details, please refer to the fit function.
            processes (int, optional):
                Number of cpu cores to use. Defaults to None.
                If None, use all available cores.
        Returns:
            dict: Pooled pairwise drift analysis results, in a shape of symmetric matrices.
        """
        ## Using ref_matrices that contains representative matrices to compare, multiprocess & call _fit function for each pair of sessions
        n_networks = len(ref_matrices)
        n_pairs = n_networks * (n_networks - 1) // 2

        ## Initialize empty arrays to store results
        pairwise_dist, pairwise_Frobenius_error, pairwise_cossim = np.zeros(
            (3, n_networks, n_networks)
        )

        ## Create args to use multiprocessing module to parallelize the computation
        iijj = itertools.combinations(
            range(n_networks), 2
        )  # all possible pairs of sessions
        args = (
            (ii, jj, ref_matrices[ii], ref_matrices[jj], weight_method, weights)
            for ii, jj in iijj
        )

        ## Multiprocessing
        if processes is None:
            processes = mp.cpu_count()

        with mp.Pool(processes=processes) as pool:
            pool_results = []
            for result in tqdm(
                pool.imap_unordered(self._pairwise_fit_batcher, args), total=n_pairs
            ):
                pool_results.append(result)

        ## Unpack results
        for ii, jj, dist, Frobenius_error, cossim in pool_results:
            pairwise_dist[ii, jj] = dist
            pairwise_Frobenius_error[ii, jj] = Frobenius_error
            pairwise_cossim[ii, jj] = cossim

        ## Create symmetric matrices
        pairwise_dist = (
            pairwise_dist + pairwise_dist.T - np.diag(np.diag(pairwise_dist))
        )
        pairwise_Frobenius_error = (
            pairwise_Frobenius_error
            + pairwise_Frobenius_error.T
            - np.diag(np.diag(pairwise_Frobenius_error))
        )
        pairwise_cossim = (
            pairwise_cossim + pairwise_cossim.T - np.diag(np.diag(pairwise_cossim))
        )

        ## Pack in a dictionary
        pairwise_results = {
            "pairwise_dist": pairwise_dist,
            "pairwise_Frobenius_error": pairwise_Frobenius_error,
            "pairwise_cossim": pairwise_cossim,
        }

        return pairwise_results
