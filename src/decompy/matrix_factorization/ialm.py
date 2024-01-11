import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult

class InexactAugmentedLagrangianMethod:
    """
        TODO:

        Reference: Augmented Lagrange multiplier method for Robust PCA - Inexact method
            - Minming Chen, October 2009. Questions? v-minmch@microsoft.com
            - Arvind Ganesh (abalasu2@illinois.edu)
    """
    def __init__(self, **kwargs) -> None:
        self.tol = kwargs.get("tol", 1e-7)
        self.maxit = kwargs.get("maxit", 1e3)
        self.maxit_inner = kwargs.get("maxit_inner", 100)
        self.verbose = kwargs.get("verbose", False)

    def decompose(
            self, 
            M: np.ndarray, 
            lambd: float = None,
            mu: float = None,
            rho: float = 1.5
        ):
        """
            lambda is the weight on sparse error term in the cost function
        """
        check_real_matrix(M)
        D = M.copy()
        m, n = D.shape
        if lambd is None:
            lambd = 1/np.sqrt(m)

        # initialize
        Y = D
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y, np.Inf) / lambd
        dual_norm = max(norm_two, norm_inf)
        Y /= dual_norm

        A_hat = np.zeros_like(D)
        E_hat = np.zeros_like(D)
        if mu is None:
            mu = 1.25 / norm_two
        mu_bar = mu * 1e7
        dnorm = np.linalg.norm(D, 'fro')

        iter = 0
        total_svd = 0
        converged = False
        sv = 10
        
        # main iteration loop
        while not converged:
            iter += 1

            temp_T = D - A_hat + (1/mu) * Y
            E_hat = np.maximum(temp_T - lambd/mu, 0) + np.minimum(temp_T + lambd/mu, 0)

            U, diagS, VT = np.linalg.svd(D - E_hat + (1/mu) * Y, full_matrices=False)
            svp = (diagS > 1/mu).sum()
            sv = min(svp + 1, n) if svp < sv else min(svp + np.round(0.05 * n), n)
            A_hat = U[:, :svp] * np.diag(diagS[:svp] - 1/mu) * VT[:svp, :]
    
            total_svd += 1
            Z = D - A_hat - E_hat

            Y += (mu * Z)
            mu = min(mu*rho, mu_bar)

            # stopping criterion
            if np.linalg.norm(Z, 'fro') / dnorm < self.tol:
                converged = True

            if self.verbose:
                print(f"Iteration: {iter}, SVD: {total_svd}, Sparse Size: {(E_hat > 0).sum()}")

            if not converged and iter >= self.maxit:
                converged = True

        return LSNResult(
            L = A_hat,
            S = E_hat,
            convergence = {
                'iteration': iter,
                'converged': (iter < self.maxit)
            }
        )
