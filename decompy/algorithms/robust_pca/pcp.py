from typing import Union
import numpy as np
import warnings

from ...utils.validations import check_real_matrix
from ...base import SVDResult

class PCP:
    """
        Principal Component Pursuit Method for Robust PCA

        Reference:
            - https://github.com/cran/rpca/blob/master/R/robustpca.R
    """

    def __init__(self, **kwargs) -> None:
        self.delta = kwargs.get("delta", 1e-7)
        self.maxiter = kwargs.get("maxiter", 5e3)
        self.verbose = kwargs.get("verbose", False)

    def threshold_l1(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def threshold_nuclear(self, M, threshold):
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        V = Vh.T
        dd = self.threshold_l1(s, threshold)
        id = np.where(dd != 0)[0]
        s = dd[id]
        U = U[:, id]
        V = V[:, id]
        L = U @ np.diag(s) @ V.T
        return (s, U, V, L)


    def decompose(self, M: np.ndarray, lambdaval: Union[float, None] = None, muval: Union[float, None] = None):
        check_real_matrix(M)
        X = np.copy(M)  # to ensure that original matrix is not modified
        n, p = X.shape
        lambdaval = lambdaval if lambdaval is not None else 1/np.sqrt(max(n, p))
        muval = muval if muval is not None else (n*p)/(4 * np.abs(X).sum())
        termnorm = self.delta * np.linalg.norm(X)

        S = np.zeros_like(X)
        Yimu = np.zeros_like(X)

        imu = 1/muval
        limu = lambdaval / muval

        niter = 0
        stats = []
        converged = False

        residnorm = 0
        while not converged:
            niter += 1
            s, U, V, L = self.threshold_nuclear(X - S + Yimu, imu)
            S = self.threshold_l1(X - L + Yimu, limu)
            MLS = X - L - S

            residnorm = np.linalg.norm(MLS)
            stats.append(residnorm)
            if self.verbose:
                print(f"Iteration: {niter}, Residual Norm: {residnorm}")
        
            converged = (residnorm < termnorm) or (niter > self.maxiter)

            Yimu += MLS
        
        finaldelta = residnorm * self.delta / termnorm
        if niter >= self.maxiter:
            warnings.warn(f"RPCA using PCP approach did not converged after {niter} iterations.\nFinal delta: {finaldelta}\nPlease consider increasing maxiter.")

        convergence_metrics = {
            'converged': converged,
            'iterations': niter,
            'finaldelta': finaldelta,
            'alldelta': np.array(stats) * (self.delta / termnorm)
        }
        return SVDResult(U, s, V, convergence = convergence_metrics)
