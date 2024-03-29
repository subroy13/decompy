from typing import Union
import numpy as np
import warnings

from ..utils.validations import check_real_matrix, is_decreasing
from ..utils.constants import EPS
from ..interfaces import SVDResult


class RobustSVDDensityPowerDivergence:
    """
    Robust SVD using Density Power Divergence based Alternating Regression Method

    Notes
    -----
    [1] Roy, Subhrajyoty, Ayanendranath Basu, and Abhik Ghosh. "A New Robust Scalable Singular Value Decomposition Algorithm for Video Surveillance Background Modelling." arXiv preprint arXiv:2109.10680 (2021).

    """

    def __init__(self, **kwargs):
        """Initialize the Robust SVD using Density Power Divergence class.

        Parameters
        ----------
        alpha : float, optional
            The alpha parameter for the robust density power divergence.
            Must be between 0 and 1. Default is 0.5.
        tol : float, optional
            Minimum tolerance level for the norm of X. Default is 1e-4.
        eps : float, optional
            Convergence criterion for change in left/right vectors.
            Default is 1e-8.
        maxiter : int, optional
            Maximum number of iterations per singular value. Default is 100.
        inneriter : int, optional
            Number of inner fixed point iterations. Default is 5.
        verbose : bool, optional
            If True, print status messages. Default is False.
        method : {'v1', 'v2'}, optional
            The method of regression. Default is 'v2'.
        initmethod : {'random', 'tsvd'}, optional
            The method for initializing vectors. Default is 'random'.

        """
        self.alpha = kwargs.get(
            "alpha", 0.5
        )  # the alpha in the robust minimum divergence
        assert (
            self.alpha >= 0 and self.alpha <= 1
        ), "alpha must be between 0 and 1"  # * Protect against absurd options
        self.tol = kwargs.get(
            "tol", 1e-4
        )  # * Minimum tolerance level for the norm of X
        self.eps = kwargs.get(
            "eps", 1e-8
        )  # * Convergence criterion for change in left / right vectors
        self.maxiter = kwargs.get(
            "maxiter", 1e2
        )  # * Maximum number of iterations per singular value
        self.inneriter = kwargs.get(
            "inneriter", 3
        )  # * Number of inner fixed point iteration
        self.verbose = kwargs.get("verbose", False)
        self.method = kwargs.get("method", "v1")  # * The method of regression
        assert self.method in ["v1", "v2"]
        self.initmethod = kwargs.get(
            "initmethod", "random"
        )  # the method of vector initialization
        assert self.initmethod in ["random", "tsvd"]

    def sanity_check(
        self, X: np.ndarray, lambdas: np.ndarray, maxit_reached: np.ndarray
    ):
        """Performs sanity checks on the output.

        Parameters
        ----------
        X : ndarray
            The input data matrix
        lambdas : ndarray
            The computed singular values
        maxit_reached : ndarray
            Boolean array indicating if max iterations were reached

        Returns
        -------
        None

        Notes
        -----
        Checks if max iterations were reached and warns if so.
        Also checks if singular values are in decreasing order.

        """
        # * Check if maxit is reached
        if maxit_reached.sum() > 0:
            warnings.warn(
                "Maximum iteration reached for singular values "
                + ",".join([str(i) for i in np.where(maxit_reached)[0] + 1]),
                stacklevel=1,
            )
        if not is_decreasing(lambdas):
            warnings.warn("Singular values are not in sorted order")

    def __decompose_v1(
        self, M: np.ndarray, rank: int, initu: np.ndarray, initv: np.ndarray
    ):
        X = np.copy(M)
        n, p = X.shape
        lambdas = np.zeros(rank)
        iterlist = np.zeros(rank)

        # normalize entries for better numerical precision
        scale_factor = np.quantile(X, [0.25, 0.75])
        X /= scale_factor[1] - scale_factor[0]
        Xnorm = np.linalg.norm(X)

        sigma = (
            np.median(np.abs(X - np.median(X))) * 1.4826
        )  # consistent estimate of sigma^2

        for r in range(rank):
            curr_relnorm = np.linalg.norm(X) / Xnorm

            if curr_relnorm > self.tol:
                niter, converged = 0, False

                while not converged:
                    curr_a = initu[:, r]
                    curr_b = initv[:, r]
                    curr_lambda = lambdas[r]

                    # STEP 1: Fix Right, Optimize Left
                    # Do fixed point iteration
                    left_iter, fixed = 0, False

                    c = np.zeros(n)
                    while not fixed:
                        curr_c = c
                        wmat = X - np.outer(c, curr_b)
                        wmat = np.exp(-0.5 * self.alpha * (wmat**2) / sigma)
                        numer = np.multiply(wmat * X, curr_b).sum(axis=1)
                        denom = np.multiply(wmat, curr_b**2).sum(axis=1)
                        c = numer / np.maximum(denom, EPS)  # since denom is always > 0
                        left_iter += 1

                        # Check if fixed point criterion is met
                        fixed = (
                            np.linalg.norm(c - curr_c) / np.linalg.norm(c) < self.eps
                        ) or (left_iter > self.inneriter)

                    # Apply Gram-Schimdt Operation
                    if r > 0:
                        c -= (initu[:, :r] @ initu[:, :r].T) @ c

                    # Normalized
                    curr_lambda = np.linalg.norm(c)
                    curr_a = c / curr_lambda if curr_lambda > EPS else np.zeros_like(c)

                    # STEP 2: Fix Left, Optimize Right
                    # Do fixed point iteration
                    right_iter, fixed = 0, False
                    d = np.zeros(p)
                    while not fixed:
                        curr_d = d
                        wmat = X - np.outer(curr_a, d)
                        wmat = np.exp(-0.5 * self.alpha * (wmat**2) / sigma)
                        numer = np.multiply((wmat * X).T, curr_a).sum(axis=1)
                        denom = np.multiply(wmat.T, curr_a**2).sum(axis=1)
                        d = numer / np.maximum(denom, EPS)
                        right_iter += 1

                        # Check if fixed point criterion is met
                        fixed = (
                            np.linalg.norm(d - curr_d) / np.linalg.norm(d) < self.eps
                        ) or (right_iter > self.inneriter)

                    # * Apply Gram-Schimdt Criterion
                    if r > 0:
                        d -= (initv[:, :r] @ initv[:, :r].T) @ d

                    # Normalize
                    curr_lambda = np.linalg.norm(d)
                    curr_b = d / curr_lambda if curr_lambda > EPS else np.zeros_like(d)

                    # STEP 3: Check if convergence is met
                    niter += 1
                    is_convl = (
                        np.abs(curr_lambda - lambdas[r])
                        / (np.abs(lambdas[r]) + self.eps)
                        < self.eps
                    )
                    is_conva = np.linalg.norm(curr_a - initu[:, r])
                    is_convb = np.linalg.norm(curr_b - initv[:, r])

                    converged = (niter > self.maxiter) or (
                        is_convl and is_conva and is_convb
                    )

                    lambdas[r], initu[:, r], initv[:, r] = curr_lambda, curr_a, curr_b

                # Outside iteration, count iteration
                iterlist[r] = niter

                # One singular value object, proceed to the next
                X -= lambdas[r] * np.outer(initu[:, r], initv[:, r])
            else:
                break

        # Change the singular values as required
        lambdas = (scale_factor[1] - scale_factor[0]) * lambdas
        initu = initu[:, :rank]  # Subset only the required column upto the rank
        initv = initv[:, :rank]

        return (lambdas, initu, initv, iterlist)

    def __decompose_v2(self, M: np.ndarray, rank: int, U: np.ndarray, V: np.ndarray):
        X = np.copy(M)
        n, p = X.shape

        # normalize entries for better numerical precision
        scale_factor = np.quantile(X, [0.25, 0.75])
        X /= scale_factor[1] - scale_factor[0]
        sigma = np.median(np.abs(X - np.median(X))) * 1.4826

        niter = 0
        sigs = np.ones(rank)  # (r, )
        sigma = np.var(X) ** 0.5  # sigma

        while True:
            # step 1: estimate V based on U
            newsigs = sigs
            for _ in range(self.inneriter):
                ermat = X - U @ np.diag(newsigs) @ V.T  # (n, p)
                wmat = np.exp(-self.alpha * ermat**2 / (2 * (sigma**2)))  # (n, p)
                for j in range(p):
                    V[j, :] = np.linalg.solve(
                        U.T @ np.diag(wmat[:, j] + EPS) @ U,
                        U.T @ np.diag(wmat[:, j]) @ X[:, j],
                    )  # (r, )

                # perform QR decomposition for gram schimdt
                V, R = np.linalg.qr(V)
                newsigs = np.absolute(np.diag(R))

            # step 2: estimate U based on V
            for _ in range(self.inneriter):
                ermat = X - U @ np.diag(newsigs) @ V.T  # (n, p)
                wmat = np.exp(-self.alpha * ermat**2 / (2 * (sigma**2)))  # (n, p)
                for i in range(n):
                    U[i, :] = np.linalg.solve(
                        V.T @ np.diag(wmat[i, :] + EPS) @ V,
                        V.T @ np.diag(wmat[i, :]) @ X[i, :],
                    )  # (r, )
                # perform QR decomposition for gram schimdt
                U, R = np.linalg.qr(U)
                newsigs = np.absolute(np.diag(R))

            # step 3: check convergence
            err = np.linalg.norm(sigs - newsigs) / np.linalg.norm(sigs)
            niter += 1
            if err < self.tol or niter >= self.maxiter:
                break
            else:
                sigs = newsigs

        # Change the singular values as required
        sigs = (scale_factor[1] - scale_factor[0]) * sigs
        return (sigs, U, V, niter)

    def _get_initialized_U_V(self, M: np.ndarray, rank):
        """The function `_get_initialized_U_V` initializes matrices U and V (orthogonal left and right singular matrices) based on the given matrix M and for a specified rank.

        Parameters
        ----------
        M : np.ndarray
            M is a numpy array representing a matrix.
        rank
            The "rank" parameter represents the desired rank of the factorization. In matrix factorization, the rank refers to the number of latent factors used to approximate the original matrix. It determines the complexity and accuracy of the factorization.

        Returns
        -------
            a tuple containing the initialized U and V matrices.

        """
        X = np.copy(M)
        n, p = X.shape
        if self.initmethod == "random":
            initu = np.random.random((n, rank))
            initv = np.random.random((p, rank))

            # make it orthogonalized
            initu, _ = np.linalg.qr(initu)
            initv, _ = np.linalg.qr(initv)
        else:
            quants = np.quantile(X, q=[0.25, 0.75])
            X[X < quants[0]] = quants[0]
            X[X > quants[1]] = quants[1]
            U, _, Vt = np.linalg.svd(X)
            initu, initv = U, Vt.T
        return (initu, initv)

    def decompose(
        self,
        M: np.ndarray,
        rank: Union[int, None] = None,
        initu: Union[np.ndarray, None] = None,
        initv: Union[np.ndarray, None] = None,
    ):
        """Decompose a matrix M into U, S, V using robust SVD.

        Parameters
        ----------
        M : ndarray
            Input matrix to decompose, of shape (n, p).
        rank : int
            Rank of the decomposition.
        initu : ndarray
            Left singular vectors at initialization, of shape (n, rank). Leave blank if to be initialized by initialization method.
        initv : ndarray
            Right singular vectors at initialization, of shape (p, rank). Leave blank if to be initialized by initialization method.

        Returns
        -------
        res: SVDResult
            A tuple containing the factorization results

            sigs : ndarray
                Singular values, of shape (rank,).
            U : ndarray
                Left singular vectors, of shape (n, rank).
            V : ndarray
                Right singular vectors, of shape (p, rank).
            niter : int
                Number of iterations taken.

        Notes
        -----
        Implements the rSVD-DPD algorithm from [1]_.

        References
        ----------
        .. [1] T. Zhou and D. Tao, "GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy Case", ICML-11.

        """
        check_real_matrix(M)
        n, p = M.shape

        # Protect against absurd options
        if rank is not None:
            rank = max(1, min(rank, min(n, p)))
        else:
            rank = min(n, p)

        # initialize initU, initV
        if initu is None or initv is None:
            initu, initv = self._get_initialized_U_V(M, rank)
        else:
            # both are present
            assert (
                initu.shape[0] == n
                and initv.shape[0] == p
                and initu.shape[1] == rank
                and initv.shape[1] == rank
            ), "Invalid initialization value"

        if self.method == "v1":
            lambdas, initu, initv, iterlist = self.__decompose_v1(M, rank, initu, initv)
        else:
            lambdas, initu, initv, iterlist = self.__decompose_v2(M, rank, initu, initv)

        idx = lambdas.argsort()[::-1]  # sort the singular values in decreasing order
        lambdas = lambdas[idx]
        initu = initu[:, idx]
        initv = initv[:, idx]  # rearrage the singular values in the same order

        if self.verbose:
            self.sanity_check(M, lambdas, (iterlist >= self.maxiter))

        return SVDResult(initu, lambdas, initv, convergence={"iterations": iterlist})
