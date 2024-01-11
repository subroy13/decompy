from typing import Union
import numpy as np

from ..interfaces import LSNResult

class SingularValueThresholding:

    """
    Robust SVD using Density Power Divergence based Alternating Regression Method

    Notes
    -----
    [1] Roy, Subhrajyoty, Ayanendranath Basu, and Abhik Ghosh. "A New Robust Scalable Singular Value Decomposition Algorithm for Video Surveillance Background Modelling." arXiv preprint arXiv:2109.10680 (2021).
    
    """

    def __init__(self, **kwargs) -> None:
        self.verbose = kwargs.get("verbose", False)
        self.maxiter = kwargs.get("maxiter", 25e3)
        self.epsilon = kwargs.get("epsilon", 5e-4)

    def decompose(self, M: np.ndarray, lambdaval: float, tau: Union[float, None] = None, delta: Union[float, None] = None):
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




