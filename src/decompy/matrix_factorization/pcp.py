from typing import Union
import numpy as np
import warnings

from ..utils.validations import check_real_matrix
from ..interfaces import SVDResult

class PrincipalComponentPursuit:
    """
    Robust PCA using Principal Component Pursuit Method

    Notes
    -----
    [1] Candès, Emmanuel J., et al. "Robust principal component analysis?." Journal of the ACM (JACM) 58.3 (2011): 1-37.

    """

    def __init__(self, **kwargs):
        """Initialize Principal Component Pursuit class.

        Parameters
        ----------
        delta : float, optional
            Regularization parameter. Default is 1e-7.
        maxiter : int, optional
            Maximum number of iterations. Default is 5000.
        verbose : bool, optional
            Whether to print progress messages. Default is False.

        """
        self.delta = kwargs.get("delta", 1e-7)
        self.maxiter = kwargs.get("maxiter", 5e3)
        self.verbose = kwargs.get("verbose", False)

    def _threshold_l1(self, x: np.ndarray, threshold: float):
        """Apply L1 thresholding to array x.
        
        Parameters
        ----------
        x : ndarray
            Input array.
        threshold : float
            Threshold value. Values with absolute value less than
            threshold are set to 0.
        
        Returns
        -------
        ndarray
            Thresholded array.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _threshold_nuclear(self, M: np.ndarray, threshold: float):
        """Perform nuclear norm thresholding on matrix M.

        Parameters
        ----------
        M : ndarray
            Input matrix to threshold.
        threshold : float
            Threshold value.

        Returns
        -------
        s : ndarray
            Singular values after thresholding.
        U : ndarray
            Left singular vectors.
        V : ndarray
            Right singular vectors.
        L : ndarray
            Thresholded matrix.

        """
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        V = Vh.T
        dd = self._threshold_l1(s, threshold)
        id = np.where(dd != 0)[0]
        s = dd[id]
        U = U[:, id]
        V = V[:, id]
        L = U @ np.diag(s) @ V.T
        return (s, U, V, L)


    def decompose(self, M: np.ndarray, lambdaval: Union[float, None] = None, muval: Union[float, None] = None):
        """Decompose a matrix into low-rank and sparse components using Principal Component Pursuit (PCP).

        Parameters
        ----------
        M : ndarray
            The input matrix to decompose.
        lambdaval : float or None, optional
            Regularization parameter for the nuclear norm. Default is 1/sqrt(max(n, p))
            where n, p are dimensions of M.
        muval : float or None, optional  
            Regularization parameter for the l1 norm. Default is (n*p)/(4 * abs(X).sum())
            where n, p are dimensions of M.

        Returns
        -------
        SVDResult
            A named tuple containing the low-rank component U, singular values s, 
            orthogonal matrix V and convergence metrics.

        """
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
            s, U, V, L = self._threshold_nuclear(X - S + Yimu, imu)
            S = self._threshold_l1(X - L + Yimu, limu)
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
