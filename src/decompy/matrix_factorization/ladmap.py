from typing import Union
import numpy as np

from decompy.interfaces.lsnresult import LSNResult


class LinearizedADMAdaptivePenalty:
    """
        Linearized Alternating Direction Method with Adaptive Penalty

        It aims to solve the LRR problem
        min |Z|_*+lambda*|E|_2,1  s.t., X = XZ+E, where lambda is a penalty parameter.

        Notes
        ------
        [1] Linearized Alternating Direction Method with Adaptive Penalty for Low-Rank Representation, Lin et al., 2011 - http://arxiv.org/abs/1109.0367
        [2] Original MATLAB code created by Risheng Liu on 05/02/2011, rsliu0705@gmail.com

    """
    def __init__(self, **kwargs):
        self.tol1 = kwargs.get('tol1', 1e-4)
        self.tol2 = kwargs.get('tol2', 1e-5)
        self.maxiter = kwargs.get('maxiter', 1e3)
        self.max_mu = kwargs.get('max_mu', 1e10)
        self.verbose = kwargs.get('verbose', False)


    def _solve_l2(self, W: np.ndarray, lambd: float):
        _, n = W.shape
        E = W.copy()  # make a copy so that we don't accidentally make changes to W
        for i in range(n):
            E[:, i] = self._solve_l2(W[:, i], lambd)
        return E


    def _solve_l1l2(self, w: np.ndarray, lambd: float):
        """
            It solves an optimization problem given by 
            min lambda |x|_2 + |x - w|_2^2
        """
        nw = np.linalg.norm(w)
        if nw > lambd:
            x = (nw - lambd) * w / nw   # a scaled version of w
        else:
            x = np.zeros_like(w)
        return x


    def decompose(self, M: np.ndarray, rho: Union[float, None] = None, lambd: Union[float, None] = None):
        rho = 1.9 if rho is None else rho
        lambd = 0.1 if lambd is None else lambd

        X = M.copy()  # make a copy of matrix so that we dont modify
        d, n = X.shape
        
        normfX = np.linalg.norm(X, 'fro')
        norm2X = np.linalg.norm(X, ord=2)
        mu = min(d, n) * self.tol2

        eta = 1.02 * norm2X**2   # eta needs to be larger than ||X||_2^2, but need not be too large.

        # Initializing optimization variables
        E = np.zeros((d, n))
        Y = np.zeros((d, n))
        Z = np.zeros((n, n))

        XZ = np.zeros((d, n))   # XZ is simply X * Z

        sv = 5
        svp = sv

        # start the main loop
        converged = False
        niter = 0

        while niter < self.maxiter:
            niter += 1

            Ek = E.copy()
            Zk = Z.copy()  # copy E and Z to compute the change in the solutions

            E = self._solve_l1l2(X - XZ + Y/mu, lambd / mu)
            M = Z + X.T @ (X - XZ - E + Y/mu) / eta
            U, S, V = np.linalg.svd(M, full_matrices=False)
            V = V.T  # we want V to be column oriented

            svp = np.sum(S > 1/(mu * eta))
            if svp < sv:
                sv = min(svp + 1, n)
            else:
                sv = min(svp + round(0.05 * n), n)
            
            if svp >= 1:
                S = S[:svp] - 1/(mu * eta)
            else:
                svp = 1
                S = np.zeros(1)
            
            A_U = U[:, :svp]
            A_s = S
            A_V = V[:svp, :]

            Z = A_U @ np.diag(A_s) @ A_V

            diffZ = np.linalg.norm(Zk - Z, 'fro')
            relChgZ = diffZ / normfX
            relChgE = np.linalg.norm(E - Ek, 'fro') / normfX
            relChg = max(relChgZ, relChgE) 

            XZ = X @ Z  # introducing XZ to avoid computing X*Z multiple times, which has O(n^3) complexity.
            dY = X - XZ - E
            recErr = np.linalg.norm(dY, 'fro') / normfX

            converged = recErr < self.tol1  and relChg < self.tol2

            if self.verbose:
                if niter == 1 or niter % 50 == 0 or converged:
                    print(
                        f"iter {iter}, mu={mu}, rank(Z)={svp}, "
                        f"relChg={max(relChgZ, relChgE)}, recErr={recErr}"
                    )
            if converged:
                break 
            else:
                Y += (mu * dY)
                if mu*relChg < self.tol2:
                    mu = min(self.max_mu, mu* rho)
                
        return LSNResult(
            L = X @ Z,
            S = E,
            N = None,
            convergence = {
                'iteration': niter,
                'converged': converged,
                'relative change': relChg,
                'reconstruction error': recErr
            }
        )
