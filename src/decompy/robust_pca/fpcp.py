from typing import Union
import numpy as np
import warnings

from ..utils.validations import check_real_matrix
from ..base import SVDResult


class FastPrincipalComponentPursuit:
    """
    Robust PCA using Fast PCP Method

    Notes
    -----
    [1] [Rodriguez and Wohlberg, 2013](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6738015)

    """

    def __init__(self, **kwargs) -> None:
        self.maxiter = kwargs.get("maxiter", 2)
        self.verbose = kwargs.get("verbose", False)

    def _shrink(self, v, lambdval):
        u = np.sign(v) * np.maximum(0, np.abs(v) - lambdval)
        return u

    def decompose(self, M: np.ndarray, initrank: Union[int, None] = None, rank_threshold: Union[float, None] = None, lambdaval: Union[float, None] = None, lambdafactor: Union[float, None] = None):
        check_real_matrix(M)
        X = np.copy(M)  # to ensure that original matrix is not modified
        n, p = X.shape
        lambdaval = lambdaval if lambdaval is not None else 1/np.sqrt(max(n, p))
        lambdafactor = 1 if lambdafactor is None else lambdafactor
        rank0 = 1 if initrank is None else initrank
        rank_threshold = 0.01 if rank_threshold is None else rank_threshold

        inc_rank = True  # a flag to increment the rank plus one at each iteration

        # First outer loop
        rank = rank0
        Ulan, Slan, Vtlan = np.linalg.svd(X)
        Ulan = Ulan[:, :rank]
        Slan = Slan[:rank]
        Vtlan = Vtlan[:rank, :]
        
        # current low rank approximation
        L1 = Ulan @ np.diag(Slan) @ Vtlan

        # shrinkage
        S1 = self._shrink(X - L1, lambdaval)

        # Outer loops
        niter = 0
        while True:
            niter += 1
            if inc_rank:
                lambdaval *= lambdafactor
                rank += 1
            
            # get the current rank estimate
            Ulan, Slan, Vtlan = np.linalg.svd(X - S1)
            Ulan = Ulan[:, :rank]
            Slan = Slan[:rank]
            Vtlan = Vtlan[:rank, :]

            # simple rule to keep or increase the current rank's value
            rho = Slan[rank-1] / np.sum(Slan[:(rank-1)])
            inc_rank = (rho > rank_threshold)

            # current low rank approximation
            L1 = Ulan @ np.diag(Slan) @ Vtlan

            # shrinkage
            S1 = self._shrink(X - L1, lambdaval)

            if not inc_rank or niter >= self.maxiter:
                break

        return SVDResult(Ulan, Slan, Vtlan.T, convergence = {
            'converged': (not inc_rank),
            'iterations': niter
        })
