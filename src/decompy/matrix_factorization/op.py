import numpy as np

from ..utils.validations import check_real_matrix, check_binary_matrix
from ..interfaces import LSNResult

class OutlierPursuit:
    """
    Matrix completion algorithm via Outlier Pursuit
    
    Notes
    -----
    [1] H. Xu, C. Caramanis and S. Sanghavi, "Robust PCA via Outlier Pursuit," in IEEE Transactions on Information Theory, vol. 58, no. 5, pp. 3047-3064, May 2012, doi: 10.1109/TIT.2011.2173156.

    """
    def __init__(self, **kwargs):
        self.full_svd = kwargs.get("full_svd", True)
        self.increaseK = kwargs.get("increaseK", 10)
        self.maxiter = kwargs.get("maxiter", 1e3)

    def _iterate_L(self, L, epsilon, starting_K):
        if not self.full_svd:
            pass
            # TODO: implement lansvd()
            # U, S, V = lansvd(L, starting_K, 'T', epsilon, self.increaseK)
            # rank_out = min(S.shape)
        else:
            U, S, Vh = np.linalg.svd(L)
            rank_out = 0
        S = np.where(S > epsilon, S - epsilon, np.where(S < -epsilon, S + epsilon, 0))
        output = U @ np.diag(S) @ Vh
        return output, rank_out
        

    def _iterate_C(self, C, epsilon):
        m, n = C.shape
        output = np.zeros_like(C)
        for i in range(n):
            temp = C[:, i]
            norm_temp = np.linalg.norm(temp)
            if norm_temp > epsilon:
                temp -= temp * epsilon / norm_temp
            else:
                temp = np.zeros(m)
            output[:, i] = temp
        return output
    

    def decompose(self, M: np.ndarray, rank: int = None, lambd: float = None, Omega_mask: np.ndarray = None):
        check_real_matrix(M)
        X = np.copy(M)
        m, n = X.shape
        if Omega_mask is not None and Omega_mask.shape != X.shape:
            raise ValueError('Size of Omega and M are inconsistent')

        # initialize the parameters
        if Omega_mask is None:
            Omega_mask = np.ones((m, n))
        if rank is None or rank < 0:
            initial_rank = np.ceil(min(m, n) * 0.1)
        else:
            initial_rank = min(m, n, rank)
        if lambd is None:
            lambd = 1/(min(m, n)**0.5)

        delta = 1e-5
        mu_temp = 0.99 * np.linalg.norm(X)
        mu_bar = delta * mu_temp
        eta = 0.9
        tol = 1e-6 * np.linalg.norm(X, "fro")
        rank_L = initial_rank

        converged = False

        # initial parameters
        # L_temp0, C_temp0 is L_k and C_k
        # L_temp1, C_temp1 is L_{k-1} and C_{k-1}
        L_temp0 = np.zeros((m, n))
        L_temp1 = np.zeros((m, n))
        C_temp0 = np.zeros((m, n))
        C_temp1 = np.zeros((m, n))
        t_temp0 = 1
        t_temp1 = 1
        niter = 0

        while not converged:
            niter += 1

            YL = L_temp0 + (t_temp1 - 1) / t_temp0 * (L_temp0 - L_temp1)
            YC = C_temp0 + (t_temp1 - 1) / t_temp0 * (C_temp0 - C_temp1)
            M_difference = (YL + YC - M) * Omega_mask
            
            GL = YL - 0.5 * M_difference
            L_new, rank_L  = self._iterate_L(GL, mu_temp/2, rank_L + 1)

            GC = YC - 0.5 * M_difference
            C_new = self._iterate_C(GC, mu_temp * lambd / 2)

            t_new = (1 + np.sqrt(4 * t_temp0 ** 2 + 1)) / 2
            mu_new = max(eta * mu_temp, mu_bar)

            # Now to decide whether to stop
            S_L = 2 * (YL - L_new) + (L_new + C_new - YL - YC)
            S_C = 2 * (YC - C_new) + (L_new + C_new - YL - YC)
            if np.linalg.norm(S_L, 'fro') ** 2 + np.linalg.norm(S_C, 'fro') ** 2 <= tol ** 2 or niter > self.maxiter:
                converged = True
            else:
                # update the parameters
                L_temp1 = L_temp0
                L_temp0 = L_new
                C_temp1 = C_temp0
                C_temp0 = C_new
                t_temp1 = t_temp0
                t_temp0 = t_new
                mu_temp = mu_new
    
        return LSNResult(
            L = L_new,
            S = C_new,
            convergence = {
                'niter': niter,
                'converged': (niter < self.maxiter)
            }
        )


