from typing import Union
import numpy as np

from ..interfaces import LSNResult

class SingularValueThresholding:
    """ Implements the Singular Value Thresholding (SVT) algorithm for 
        Robust PCA.
    """

    def __init__(self, **kwargs):
        """Initialize the SVT class.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is False.

        maxiter : float, optional
            Maximum number of iterations. Default is 25000.

        epsilon : float, optional
            Tolerance for stopping criterion. Default is 0.0005.
        """
        self.verbose = kwargs.get("verbose", False)
        self.maxiter = kwargs.get("maxiter", 25e3)
        self.epsilon = kwargs.get("epsilon", 5e-4)

    def decompose(self, M: np.ndarray, lambdaval: float, tau: Union[float, None] = None, delta: Union[float, None] = None):
        """Decompose a matrix M into low-rank (L) and sparse (S) components.

        Parameters
        ----------
        M : ndarray
            Input matrix to decompose
        lambdaval : float
            Regularization parameter for sparse component
        tau : float or None, optional
            Threshold for singular values, by default None
        delta : float or None, optional
            Step size for dual ascent, by default None

        Returns
        -------
        LSNResult
            Named tuple containing low-rank matrix L, sparse matrix S, 
            noise matrix N, and convergence info
        """
        X = np.copy(M)
        n, p = X.shape

        # set options
        delta = 0.9 if delta is None else delta
        tau = 1e4 if tau is None else tau

        Y = np.zeros((n, p))  # lagrangian multiplier
        A = np.zeros((n, p))  # structure
        E = np.zeros((n, p))  # error

        niter = 0
        rankA = 0
        converged = False 

        while not converged:
            niter += 1
            U, s, Vt = np.linalg.svd(Y)
            U = U[:, :rankA]
            Vt = Vt[:rankA, :]

            A = U @ np.diag(np.maximum(s - tau, 0)) @ Vt
            E = np.sign(Y) * np.maximum(np.abs(Y) - lambdaval * tau)
            M2 = M - A - E

            rankA = np.sum(s > tau)  # approx rank of A
            cardE = np.sum(np.abs(E) > 0)  # approx number of nonzero entries in sparse component

            Y = Y + delta * M2

            if np.linalg.norm(M2) / np.linalg.norm(M) < self.epsilon or niter >= self.maxiter:
                converged = True

        if niter >= self.maxiter and self.verbose:
            print(f"Maximum number of iterations reached")

        return LSNResult(
            L = A,
            S = E,
            N = None,
            convergence = {
                'converged': (niter < self.maxiter),
                'iterations': niter
            }
        )     




