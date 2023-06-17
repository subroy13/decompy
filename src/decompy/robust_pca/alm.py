import numpy as np

from ..utils.validations import check_real_matrix
from ..base import LSNResult

class AugmentedLagrangianMethod:
    """
        Reference: Robust principal component analysis based on low-rank and block-sparse matrix decomposition
            - Gongguo Tang; Arye Nehorai
            - Link: https://ieeexplore.ieee.org/document/5766144
    """

    def __init__(self, **kwargs) -> None:
        self.tol_inner1 = kwargs.get("tol_inner1", 1e-4)
        self.tol_inner2 = kwargs.get("tol_inner2", 1e-6)
        self.tol_out = kwargs.get("tol", 1e-7)
        self.maxiter_inner1 = kwargs.get("maxiter_inner1", 1)
        self.maxiter_inner2 = kwargs.get("maxiter_inner2", 20)
        self.maxiter_out = kwargs.get("maxiter", 500)
        self.verbose = kwargs.get("verbose", False)

        self.alpha = kwargs.get("alpha", 1)
        self.beta = kwargs.get("beta", 0.2)
        self.rho = kwargs.get("rho", 1.1)

    def decompose(self, M: np.ndarray, rank: int = None, kappa: float = None, tau: float = None):
        check_real_matrix(M)
        D = M.copy()
        n, p = D.shape

        # Initialization
        kappa = 1.1 if kappa is None else kappa
        tau = 0.61 if tau is None else tau
        lambd = tau * kappa
        eta = (1 - tau)* kappa
        mu = 30 / np.linalg.norm(np.sign(D))

        Y = np.zeros_like(D)
        E = np.zeros_like(D)
        A = D

        iter_out = 0
        err_out = 1
        sv = 10 if rank is None else rank
        tol_iter = 0

        # main iteration loop (outer)
        while iter_out < self.maxiter_out and err_out > self.tol_out:
            iter_out += 1

            Ak, Ek = A, E
            iter_inner1 = 0
            err_inner1 = 1

            # inner iteration loop - 1
            while iter_inner1 < self.maxiter_inner1 and err_inner1 > self.tol_inner1:
                iter_inner1 += 1

                G = D - Ek + Y/mu
                Akk = G
                Ahk = np.zeros_like(Akk)

                iter_inner2 = 0
                err_inner2 = 1

                while iter_inner2 < self.maxiter_inner2 and err_inner2 > self.tol_inner2:
                    iter_inner2 += 1

                    U, diagS, VT = np.linalg.svd(Akk, full_matrices=False)
                    diagS = diagS[:sv]   # only take sv many vectors
                    svn = np.sum(diagS > self.beta)
                    svp = svn

                    ratio = diagS[:-1] / diagS[1:]
                    max_idx = np.argmax(ratio)
                    max_ratio = ratio[max_idx]

                    if max_ratio > 2:
                        svp = min(svn, max_idx)
                    if svp < sv:
                        sv = min(svp + 1, n)
                    else:
                        sv = min(svp + 10, n)

                    Ahk = U[:, :svp] @ np.diag(diagS[:svp] - self.beta) @ VT[:svp, :]

                    B = 2 * Ahk - Akk + mu * self.beta * G
                    ns = np.linalg.norm(B)
                    B = np.multiply(B / (1 + mu * self.beta), np.maximum(0, 1 - self.beta * eta / ns))
                    Akk += (self.alpha * (B - Ahk))
                    err_inner2 = self.alpha * np.linalg.norm(B - Ahk, 'fro')
                    
                    tol_iter += 1
                
                G = D - Ahk + Y/mu
                ns = np.linalg.norm(G)
                Ep = np.multiply(G, np.maximum(0, 1 - lambd / (mu * ns) ) )

                err_inner1 = max(np.linalg.norm(Ek - Ep, "fro"), np.linalg.norm(Ak - Ahk, "fro") )
                Ek = Ep
                Ak = Ahk 

            A, E = Ak, Ek 
            err_out = np.linalg.norm(D - A - E, 'fro') / np.linalg.norm(D, 'fro')
            Y += (mu * (D - A - E))
            mu *= self.rho 

        return LSNResult(
            L = A,
            S = E,
            N = None,
            convergence = {
                'niter': iter_out,
                'converged': (iter_out < self.maxiter_out)
            }
        )








                

            
