from typing import Union
import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import SVDResult


class FastPrincipalComponentPursuit:
    """
    Robust PCA using Fast Principal Component Pursuit Method

    Notes
    -----
    [1] P. Rodríguez and B. Wohlberg, "Fast principal component pursuit via alternating minimization," 2013 IEEE International Conference on Image Processing, Melbourne, VIC, Australia, 2013, pp. 69-73, doi: 10.1109/ICIP.2013.6738015. 
    """

    def __init__(self, **kwargs):
        """Initialize the FPCP algorithm.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Default is 2. 
        verbose : bool, optional
            Whether to print status messages during solving. Default is False.

        """
        self.maxiter = kwargs.get("maxiter", 2)
        self.verbose = kwargs.get("verbose", False)

    def _shrink(self, v, lambdval):
        """Shrinks values in v towards zero.
    
        This applies soft thresholding/shrinkage to each value in v. 
        Values that are less than the lambdval are set to zero.
        
        Parameters
        ----------
        v : numpy array
            Array to apply shrinkage to.
        lambdval : float
            Shrinkage parameter. Values in v less than this are set to zero.
            
        Returns
        -------
        u : numpy array
            Shrunk version of v.
        """
        u = np.sign(v) * np.maximum(0, np.abs(v) - lambdval)
        return u

    def decompose(self, M: np.ndarray, initrank: Union[int, None] = None, rank_threshold: Union[float, None] = None, lambdaval: Union[float, None] = None, lambdafactor: Union[float, None] = None):
        """Decompose a matrix M using FPCP method.

        Parameters
        ----------
        M : ndarray
            The input matrix to decompose.
        initrank : int or None, optional
            The initial rank estimate. If None, set to 1.
            Default is None.
        rank_threshold : float or None, optional
            The threshold value for incrementing the rank. 
            If None, set to 0.01.
            Default is None.
        lambdaval : float or None, optional
            The regularization parameter. If None, set to 1/sqrt(max(n,p)).
            Default is None.
        lambdafactor : float or None, optional
            The factor to decrease lambda at each iteration. 
            If None, set to 1.
            Default is None.

        Returns
        -------
        SVDResult
            A named tuple containing the final U, S, V^T matrices along
            with convergence information.
        """
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
