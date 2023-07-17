"""
Customed version of NetRep from Alex Williams' github repo: https://github.com/ahwillia/netrep
Credits to Alex Williams for the original code.
"""

from __future__ import annotations
import itertools
import multiprocessing
from typing import Literal, Tuple, Optional, List

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator
from tqdm import tqdm


class LinearMetric(BaseEstimator):
    """Computes distance between two sets of optimally linearly aligned representations.
    """

    def __init__(
            self, 
            alpha: float = 1.0, 
            center_columns: bool = True, 
            zero_pad: bool = True,
            score_method: Literal["angular", "euclidean"] = "angular"
        ):
        """
        Parameters
        ----------
        alpha : float
            Regularization parameter between zero and one. When
            (alpha == 1.0) the metric only allows for rotational
            alignments. When (alpha == 0.0) the metric allows for
            any invertible linear transformation.

        center_columns : bool
            If True, learn a mean-centering operation in addition
            to the linear/rotational alignment.

        zero_pad : bool
            If False, an error is thrown if representations are
            provided with different dimensions. If True, the smaller
            matrix is zero-padded prior to allow for an alignment.
            Some amount of regularization (alpha > 0) is required to
            align zero-padded representations.

        score_method : {'angular','euclidean'}, default='angular'
            String specifying ground metric.
        """

        if (alpha > 1) or (alpha < 0):
            raise ValueError(
                "Regularization parameter `alpha` must be between zero and one.")

        if score_method not in ("euclidean", "angular"):
            raise ValueError(
                "Expected `score_method` parameter to be in {'angular','euclidean'}. " +
                f"Found instead score_method == '{score_method}'."
            )

        self.alpha = alpha
        self.center_columns = center_columns
        self.zero_pad = zero_pad
        self.score_method = score_method

    def partial_fit(
        self, 
        X: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Computes partial whitening transformation for a response matrix."""
        if self.center_columns:
            mx = np.mean(X, axis=0)
            Xw, Zx = whiten(X - mx[None, :], self.alpha, preserve_variance=True)
        else:
            mx = np.zeros(X.shape[1])
            Xw, Zx = whiten(X, self.alpha, preserve_variance=True)
        return (mx, Xw, Zx)

    def finalize_fit(
        self, 
        cache_X: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        cache_Y: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        ) -> LinearMetric:
        """
        Takes outputs of 'partial_fit' function and finishes fitting
        transformation matrices (Wx, Wy) and bias terms (mx, my) to
        align a pair of neural activations.
        """

        # Extract whitened representations.
        self.mx_, Xw, Zx = cache_X
        self.my_, Yw, Zy = cache_Y

        # Fit optimal rotational alignment.
        U, _, Vt = np.linalg.svd(Xw.T @ Yw)
        self.Wx_ = Zx @ U
        self.Wy_ = Zy @ Vt.T

        ## Extract partial whitening matrices.
        self.Zx_ = Zx
        self.Zy_ = Zy

        ## Extract rotation matrices after partial whitening.
        self.Rx_ = U
        self.Ry_ = Vt.T

        return self

    def fit(self, X: npt.NDArray, Y: npt.NDArray) -> LinearMetric:
        """Fits transformation matrices (Wx, Wy) and bias terms (mx, my)
        to align a pair of neural activation matrices.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        """
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        return self.finalize_fit(
            self.partial_fit(X),
            self.partial_fit(Y)
        )

    def transform(
        self, 
        X: npt.NDArray, 
        Y: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Applies linear alignment transformations to X and Y.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        tX : ndarray
            Transformed version of X.
        tY : ndarray
            Transformed version of Y.
        """
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        return self._transform_X(X), self._transform_Y(Y)

    def fit_score(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """Fits alignment by calling `fit(X, Y)` and then evaluates
        the distance by calling `score(X, Y)`.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.
        
        Returns
        -------
        dist : float
            Distance between X and Y.
        """
        return self.fit(X, Y).score(X, Y)

    def score(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        """Computes the angular distance between X and Y in
        the aligned space.

        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons) matrix of activations.

        Returns
        -------
        dist : float
            Distance between X and Y.
        """
        if self.score_method == "angular":
            return angular_distance(*self.transform(X, Y))
        else: # self.score_method == "euclidean":
            return np.linalg.norm(
                np.subtract(*self.transform(X, Y)), ord="fro"
            )

    def _transform_X(self, X: npt.NDArray) -> npt.NDArray:
        """Transform X into the aligned space."""
        check_is_fitted(self, attributes=["Wx_"])
        if (X.shape[1] != self.Wx_.shape[0]):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(X)))
        if self.center_columns:
            return (X - self.mx_[None, :]) @ self.Wx_
        else:
            return (X @ self.Wx_)

    def _transform_Y(self, Y: npt.NDArray) -> npt.NDArray:
        """Transform X into the aligned space."""
        check_is_fitted(self, attributes=["Wy_"])
        if (Y.shape[1] != self.Wy_.shape[0]):
            raise ValueError(
                "Array with wrong shape passed to transform."
                "Expected matrix with {} columns, but got array"
                "with shape {}.".format(np.shape(Y)))
        if self.center_columns:
            return (Y - self.my_[None, :]) @ self.Wy_
        else:
            return Y @ self.Wy_

    def _compute_distance(self, i, j, X, Y, X_test, Y_test):
        """Helper function for multiprocessing."""

        self.fit(X, Y)
        dist_train = self.score(X, Y)
        if X_test is None and Y_test is None:
            dist_test = np.inf
        else: 
            dist_test = self.score(X_test, Y_test)
        return i, j, dist_train, dist_test

    def _compute_distance_star(self, args):
        """Helper function for multiprocessing.
        Using this allows us to use tqdm to track progress via imap_unordered.
        """
        return self._compute_distance(*args)

    def pairwise_distances(
            self, 
            train_data: List[Tuple[npt.NDArray, npt.NDArray]], 
            test_data: Optional[List[Tuple[npt.NDArray, npt.NDArray]]]=None, 
            processes: Optional[int] = None,
            verbose: bool = True,
            ):
        """Computes pairwise distances between all pairs of networks w/ multiprocessing.

        We suggest setting "OMP_NUM_THREADS=1" in your environment variables to avoid oversubscription 
        (multiprocesses competing for the same CPU).

        Parameters
        ----------
        train_data:  List[npt.NDArray]
            List of Size([images, neurons]) for train data.
        test_data: List[npt.NDArray], optional
            List of Size([images, neurons]) for test data. If None, the output
            distance matrix will be np.inf.
        enable_caching: bool
            Whether to cache pre-transformed data.
        processes: int, optional
            Number of processes to use. If None, defaults to number of CPUs.
        verbose: bool, optional
            Whether to display progress bar.
        
        Returns
        -------
        D_train: npt.NDArray
            n_networks x n_networks distance matrix.
        D_test: npt.NDArray
            n_networks x n_networks distance matrix. If test_data is None, this is
            a matrix of np.inf.
        """
        n_networks = len(train_data)
        n_dists = n_networks*(n_networks-1)//2

        # create generator of args for multiprocessing
        ij = itertools.combinations(range(n_networks), 2)
        if test_data is None:
            args = ((i, j, train_data[i], train_data[j], None, None) for i, j in ij)
        else:
            args = ((i, j, train_data[i], train_data[j], test_data[i], test_data[j]) for i, j in ij)

        if verbose:
            print(f"Parallelizing {n_dists} distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes.")
            pbar = lambda x: tqdm(x, total=n_dists, desc="Computing distances")
        else:
            pbar = lambda x: x

        with multiprocessing.Pool(processes=processes) as pool:
            results = []
            for result in pbar(pool.imap_unordered(self._compute_distance_star, args)):
                results.append(result)
       
        D_train = np.zeros((n_networks, n_networks))
        D_test = np.zeros((n_networks, n_networks))

        for i, j, dist_train, dist_test in results:
            D_train[i, j], D_train[j, i] = dist_train, dist_train
            D_test[i, j], D_test[j, i] = dist_test, dist_test

        return D_train, D_test
    

def whiten(
    X: npt.NDArray, 
    alpha: float, 
    preserve_variance: bool = True, 
    eigval_tol=1e-7
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return regularized whitening transform for a matrix X.

    Parameters
    ----------
    X : ndarray
        Matrix with shape `(m, n)` holding `m` observations
        in `n`-dimensional feature space. Columns of `X` are
        expected to be mean-centered so that `X.T @ X` is
        the covariance matrix.
    alpha : float
        Regularization parameter, `0 <= alpha <= 1`. When
        `alpha == 0`, the data matrix is fully whitened.
        When `alpha == 1` the data matrix is not transformed
        (`Z == eye(X.shape[1])`).
    preserve_variance : bool
        If True, rescale the (partial) whitening matrix so
        that the total variance, trace(X.T @ X), is preserved.
    eigval_tol : float
        Eigenvalues of covariance matrix are clipped to this
        minimum value.

    Returns
    -------
    X_whitened : ndarray
        Transformed data matrix.
    Z : ndarray
        Matrix implementing the whitening transformation.
        `X_whitened = X @ Z`.
    """

    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, np.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = np.linalg.eigh(X.T @ X)
    lam = np.maximum(lam, eigval_tol)

    # Compute diagonal of (partial) whitening matrix.
    # 
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:

        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = np.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * np.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= np.sqrt(np.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    return X @ Z, Z

def angular_distance(X: npt.NDArray, Y: npt.NDArray) -> float:
    """Computes angular distance based on Frobenius inner product between two matrices.

    Parameters
    ----------
    X : (m x n) ndarray
    Y : (m x n) ndarray

    Returns
    -------
    distance : float between zero and pi.
    """
    normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
    corr = np.dot(X.ravel(), Y.ravel()) / normalizer
    # numerical precision issues require us to clip inputs to arccos
    return np.arccos(np.clip(corr, -1.0, 1.0))

def check_equal_shapes(
    X: npt.NDArray, 
    Y: npt.NDArray, 
    nd: int = 2, 
    zero_pad: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Checks that X and Y have equal shapes."""

    X = check_array(X, allow_nd=True)
    Y = check_array(Y, allow_nd=True)

    if (X.ndim != nd) or (Y.ndim != nd):
        raise ValueError(
            "Expected {}d arrays, but shapes were {} and "
            "{}.".format(nd, X.shape, Y.shape)
        )

    if X.shape != Y.shape:

        if zero_pad and (X.shape[:-1] == Y.shape[:-1]):
            
            # Number of padded zeros to add.
            n = max(X.shape[-1], Y.shape[-1])
            
            # Padding specifications for X and Y.
            px = np.zeros((nd, 2), dtype="int")
            py = np.zeros((nd, 2), dtype="int")
            px[-1, -1] = n - X.shape[-1]
            py[-1, -1] = n - Y.shape[-1]

            # Pad X and Y with zeros along final axis.
            X = np.pad(X, px)
            Y = np.pad(Y, py)

        else:
            raise ValueError(
                "Expected arrays with equal dimensions, "
                "but got arrays with shapes {} and {}."
                "".format(X.shape, Y.shape))

    return X, Y
