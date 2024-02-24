from typing import Union
import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult


class ActiveSubspaceRobustPCA:
    """
    AS-RPCA: Active Subspace: Towards Scalable Low-Rank Learning

    Notes
    -----
    [1] Guangcan Liu and Shuicheng Yan. 2012. Active subspace: Toward scalable low-rank learning. Neural Comput. 24, 12 (December 2012), 3371-3394. https://doi.org/10.1162/NECO_a_00369

    """
    def __init__(self, **kwargs):
        """Initialize Active Subspace Robust PCA object.

        Parameters
        ----------
        maxiter : float, optional
            Maximum number of iterations. Default is 1e3.
        max_mu : float, optional
            Maximum value for mu. Default is 1e10.
        tol : float, optional
            Tolerance for stopping criteria. Default is 1e-8.
        rho : float, optional
            Factor for increasing mu. Default is 1.1.
        """
        self.maxiter = kwargs.get('maxiter', 1e3)
        self.max_mu = kwargs.get('max_mu', 1e10)
        self.tol = kwargs.get('tol', 1e-8)
        self.rho = kwargs.get('rho', 1.1)

    def decompose(self, M: np.ndarray, k: Union[int, None] = None, lambd = None):
        """Decompose a matrix into low rank and sparse components.
        
        Decomposes the input matrix `M` into the sum of 
        a low-rank matrix `L` and a sparse matrix `S`,  
        by solving the optimization problem:
        
        min |L|_* + lambda |S|_1  
        s.t. M = L + S
        
        Parameters
        ----------
        M : ndarray
            Input matrix to decompose
            
        k : int or None, optional
            Rank of the low-rank component. 
            If None, default is max(1, round(min(M.shape)/10)) 
            
        lambd : float, optional
            Regularization parameter for the sparsity term.
            If None, default is 1/sqrt(min(M.shape))
            
        Returns
        -------
        LSNResult
            Named tuple containing:
            
            L : Low-rank component
            
            S : Sparse component
            
            N : Null component (all zeros)
            
            convergence : Dictionary with keys:
                'niter' : Number of iterations
                'converged' : Whether converged within max iterations
                'final_error' : Final reconstruction error
                
        """
        check_real_matrix(M)
        X = M.copy()   # create a copy to avoid modifying true matrix

        d, n = X.shape
        if lambd is None:
            lambd = 1/np.sqrt(min(d, n))
        if k is None:
            k = max(1, np.round(min(d, n) / 10))

        tol = self.tol * np.linalg.norm(X, 'fro')
        mu = 1.0 / np.linalg.norm(X, 2)

        # initialize optimization variables
        J = np.zeros((k, n))
        E = np.zeros((d, n))
        Y = np.zeros((d, n))

        # start the main loop
        niter = 0
        while niter < self.maxiter:
            niter += 1
            dey = X - E + Y/mu
            temp = dey @ J.T

            # update Q
            U, sigma, Vt = np.linalg.svd(temp)
            Q = U @ Vt

            # update J
            temp = Q.T @ dey
            U, sigma, Vt = np.linalg.svd(temp)
            svp = (sigma > 1/mu)   # this is boolean array
            if np.sum(svp) > 1:
                sigma = sigma[svp] - 1/mu
                J = U[:, svp] @ np.diag(sigma) @ Vt[svp, :]
            else:
                svp, sigma = 1, 0
                J = np.zeros((k, n))

            # update E
            A = Q @ J.T
            temp = X - A + Y/mu
            E = np.maximum(0, temp - lambd/mu) + np.minimum(0, temp + lambd/mu)

            leq = X - A - E    # the left out part
            stop_c = np.linalg.norm(leq, 'fro')
            if stop_c < tol:
                break
            else:
                Y += (mu * leq)
                mu = min(self.max_mu, mu * self.rho)

        return LSNResult(
            L = A,
            S = E,
            N = None,
            convergence = {
                'niter': niter,
                'converged': (niter < self.maxiter),
                'final_error': stop_c
            }
        )
        