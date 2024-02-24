from typing import Union
import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult

class AlternatingDirectionMethod:
    """
    Sparse and low-rank matrix decomposition using Alternating Direction Methods

    Notes
    -----
    [1] Yuan, Xiaoming, and Junfeng Yang. "Sparse and low-rank matrix decomposition via alternating direction methods." preprint 12.2 (2009).
    
    """
    
    def __init__(self, **kwargs):
        """Initialize Alternating Direction Method matrix factorization model.

        Parameters
        ----------
        tol : float, optional
            Tolerance for stopping criteria (default is 1e-6). 
        maxiter : int, optional
            Maximum number of iterations (default is 1000).
        verbose : bool, optional
            Whether to print progress messages (default is False).

        """
        self.tol = kwargs.get('tol', 1e-6)
        self.maxiter = kwargs.get("maxiter", 1e3)
        self.verbose = kwargs.get("verbose", False)

    def decompose(
            self, 
            M: np.ndarray, 
            tau: float = 0.1,
            beta: Union[float, None] = None,
            initA: Union[np.ndarray, None] = None,
            initB: Union[np.ndarray, None] = None,
            initLambda: Union[np.ndarray, None] = None
        ):
        """Decompose a matrix M into a low-rank and sparse component using ADM.

        Parameters
        ----------
        M : ndarray
            The input matrix to decompose.
        tau : float, optional
            Thresholding parameter. Default is 0.1.
        beta : float or None, optional
            Regularization parameter. Default is 0.25 / abs(M).mean().
        initA : ndarray or None, optional
            Initial guess for the sparse component. Default is zeros.
        initB : ndarray or None, optional 
            Initial guess for the low-rank component. Default is zeros.
        initLambda : ndarray or None, optional
            Initial guess for the Lagrange multiplier. Default is zeros.

        Returns
        -------
        LSNResult
            A named tuple containing the low-rank matrix L, sparse matrix S, 
            and convergence info.

        """
        check_real_matrix(M)
        C = np.copy(M)
        if beta is None:
            beta: float = 0.25 / np.abs(C).mean()
        
        # initialization
        m, n = C.shape
        A = np.zeros_like(C) if initA is None else initA
        B = np.zeros_like(C) if initB is None else initB
        Lambda = np.zeros_like(C) if initLambda is None else initLambda
        

        # main iteration loop
        for it in range(1, int(self.maxiter) + 1):
            nrmAB = np.linalg.norm(np.hstack((A, B)), "fro")

            # A - subproblem
            X = Lambda / beta + C
            Y = X - B
            dA = A
            A = np.sign(Y) * np.maximum(0, np.abs(Y) - tau / beta)
            dA = A - dA

            # B - subproblem
            Y = X - A
            dB = B
            U, D, VT = np.linalg.svd(Y, full_matrices=False)
            ind = (D > 1/beta)
            D = np.diag(D[ind] - 1 / beta)
            B = U[:, ind] @ D @ VT[ind, :]
            dB = B - dB

            # stopping criterion
            rel_chg = np.linalg.norm(np.hstack((dA, dB)), 'fro') / (1 + nrmAB)
            if self.verbose:
                print(f"Iteration: {it}, Relative Change: {rel_chg:.2f}")
            if rel_chg < self.tol:
                break

            # Update lambda 
            Lambda -= (beta * (A + B - C))

        return LSNResult(
            L = B,
            S = A,
            convergence = {
                'niter': it,
                'convergence': (it <= self.maxiter)
            }
        )