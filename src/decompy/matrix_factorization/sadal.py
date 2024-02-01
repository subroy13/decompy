from typing import Union
import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult

class SymmetricAlternatingDirectionALM:
    """
    This is a symmetric version of the Alternating Direction Method of Multipliers (ADMM)

    Notes
    -----
    [1] ''Fast Alternating Linearization Methods for Minimizing the Sum of Two Convex Functions'', Donald Goldfarb, Shiqian Ma and Katya Scheinberg, Tech. Report, Columbia University, 2009 - 2010. 
    """
    def __init__(self, **kwargs) -> None:
        self.maxiter = kwargs.get('maxiter', 1000)
        self.sigma = kwargs.get('sigma', 1e-6) 
        self.eps = kwargs.get('eps', 1e-7)
        self.muf = kwargs.get('muf', 1e-6)
        self.sigmaf = kwargs.get('sigmaf', 1e-6)
        self.eta_mu = kwargs.get('eta_mu', 2/3)
        self.eta_sigma = kwargs.get('eta_sigma', 2/3)

    def decompose(self, M: np.ndarray, sv: Union[int, None] = None, mu: float = 0, rho: float = 0):
        check_real_matrix(M)
        D = M.copy()  # copy the original matrix so that to ensure non modification
        
        # initialization
        m, n = D.shape 
        if sv is None:
            sv = np.round(0.1 * min(m, n)) + 1
        if mu is None or mu == 0:
            mu = np.linalg.norm(D) / 1.25
        if rho is None or rho == 0:
            rho = 1/np.sqrt(n)

       
        Y = np.zeros((m, n))
        X = D - Y
        Dnorm = np.linalg.norm(D, 'fro')
        lambd = np.zeros((m, n))

        for niter in range(self.maxiter):
            U, gamma, Vt = np.linalg.svd(mu * lambd - Y + D, full_matrices=False)
            gamma_new = gamma - mu * gamma / np.maximum(gamma, mu + self.sigma) 
            svp = (gamma > mu).sum()
            if svp < sv:
                sv = min(svp + 1, n)
            else:
                sv = min(svp + np.round(0.05 * n), n)
            
            X = U @ np.diag(gamma_new) @ Vt
            lambd = lambd - (X + Y - D)/ mu
            muY = mu
            B = lambd - (X - D)/muY
            Y = muY * B - muY * np.clip(muY * B / (self.sigma + muY), -rho, rho)
            lambd -= ((X + Y - D) / muY)

            # check stopping criterion
            stop_crit = np.linalg.norm(D - X - Y, 'fro') / Dnorm
            mu = max(self.muf, mu * self.eta_mu)
            sigma = max(self.sigmaf, sigma * self.eta_sigma)

            if stop_crit < self.eps:
                break

        return LSNResult(
            L = X,
            S = Y,
            N = None,
            convergence = {
                'niter': niter,
                'converged': (niter < self.maxiter),
                'final_error': stop_crit,
            }
        ) 




