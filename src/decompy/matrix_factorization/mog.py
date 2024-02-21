import numpy as np
from scipy.special import psi, logsumexp

from ..utils.validations import check_real_matrix
from ..interfaces.lsnresult import LSNResult

class LRPrior:

    def __init__(self, a0 = None, b0 = None):
        self.a0 = a0 or 1e-6
        self.b0 = b0 or 1e-6

class MOGPrior:

    def __init__(self, mu0 = None, c0 = None, d0 = None, alpha0 = None, beta0 = None):
        self.mu0 = mu0 or 0
        self.c0 = c0 or 1e-6
        self.d0 = d0 or 1e-6
        self.alpha0 = alpha0 or 1e-6
        self.beta0 = beta0 or 1e-6

class LRModel:

    def __init__(self, U: np.ndarray, V: np.ndarray, Sigma_U: np.ndarray, Sigma_V: np.ndarray, gammas: np.ndarray):
        self.U = U
        self.V = V
        self.sigma_U = Sigma_U
        self.sigma_V = Sigma_V
        self.gammas = gammas

class MOGModel:

    def __init__(self, R: np.ndarray, c: np.ndarray, beta: np.ndarray, d: np.ndarray, mu: np.ndarray):
        self.R = R
        self.c = c
        self.beta = beta
        self.d = d
        self.mu = mu
        


class MixtureOfGaussianRobustPCA:
    """
        Original Author:
        Written by Qian Zhao (if you have any questions/comments/suggestions, please contact me: timmy.zhaoqian@gmail.com)

        References
        ----------
        [1] "Qian Zhao, Deyu Meng, Zongben Xu, Wangmeng Zuo, Lei Zhang. Robust Principal Component Analysis with Complex Noise. ICML, 2014."
    """

    def __init__(self, **kwargs):
        self.maxiter = kwargs.get("maxiter", 100)
        self.tol = kwargs.get("tol", 1e-4)
        self.mog_k = kwargs.get("mog_k", 3)
        self.init_method = kwargs.get("init_method", "SVD")
        assert self.init_method in ["SVD", "random"], "init_method must be either SVD or random"

    def __R_initialization(self, X: np.ndarray, k):
        n = X.shape[0]
        idx = np.random.choice(n, k, replace=False)
        m = X[:, idx]
        similarity = m.T @ X - np.sum(m**2, axis=0)[:, np.newaxis] / 2
        label = np.argmax(similarity, axis=0)
        _, label = np.unique(label, return_inverse=True)
        while k != len(np.unique(label)):
            idx = np.random.choice(n, k, replace=False)
            m = X[:, idx]
            similarity = m.T @ X - np.sum(m**2, axis=0)[:, np.newaxis] / 2
            label = np.argmax(similarity, axis=0)
            _, label = np.unique(label, return_inverse=True)
        R = np.zeros((n, k))
        R[np.arange(n), label] = 1  # (n, k)
        return R
    
    def __lr_update(self, Y: np.ndarray, lr_model: LRModel, mog_model: MOGModel, r: int, lr_prior: LRPrior):
        m, n = Y.shape

        a0 = lr_prior.a0
        b0 = lr_prior.b0

        U = lr_model.U
        V = lr_model.V
        Sigma_U = lr_model.sigma_U
        Sigma_V = lr_model.sigma_V
        gammas = lr_model.gammas

        R = mog_model.R
        c = mog_model.c
        d = mog_model.d
        mu = mog_model.mu

        k = mu.shape[0]

        tau = c / d  # (k, )
        Gam = np.diag(gammas)
        Rtau = (R.reshape(m*n, k) @ tau).reshape(m, n)  # (m, n)
        Rtaumu = (R.reshape(m*n, k) @ (tau * mu)).reshape(m, n)  # (m, n)
        RtauYmu = Rtau * Y - Rtaumu  # (m, n)

        # update U
        diagsU = np.zeros(r)
        tempU = np.zeros((r, r, m))
        for i in range(m):
            s1inv = (Sigma_V.reshape(r * r, n) @ Rtau[i, :]).reshape(r, r)
            s2inv = (V.T * Rtau[i, :]) @ V + Gam
            Sigma_U[:,:,i] = np.linalg.inv(s1inv + s2inv)
            U[i, :] = RtauYmu[i,:] @ V + Sigma_U[:,:,i]
            diagsU = diagsU + np.diag(Sigma_U[:,:,i])
            tempU[:, :, i] = Sigma_U[:,:,i] + U[i, :].T @ U[i, :]

        # update V
        diagsV = np.zeros(r)
        tempV = np.zeros((r, r, n))
        for j in range(n):
            s1inv = (Sigma_U.reshape(r * r, m) @ Rtau[:, j]).reshape(r, r)
            s2inv = (U.T * Rtau[:, j].T) @ U + Gam
            Sigma_V[:,:,j] = np.linalg.inv(s1inv + s2inv)
            V[j, :] = RtauYmu[:, j] @ U + Sigma_V[:,:,j]
            diagsV = diagsV + np.diag(Sigma_V[:,:,j])
            tempV[:, :, j] = Sigma_V[:,:,j] + V[j, :].T @ V[j, :]

        # update gammas
        gammas = (2 * a0 + m + n) / (2 * b0 + np.sum(U**2, axis = 0) + diagsU + np.sum(V**2, axis = 0) + diagsV)

        # prune redundant dimensions
        dim_thr = 1e2
        max_gamma = np.min(gammas) * dim_thr
        if np.sum(gammas > max_gamma) > 0:
            index = gammas > max_gamma
            U = U[:, index]
            V = V[:, index]
            gammas = gammas[index]
            Sigma_U = Sigma_U[index, index, :]
            Sigma_V = Sigma_V[index, index, :]
            tempU = tempU[index, index, :]
            tempV = tempV[index, index, :]
            r = U.shape[1]

        lr_model = LRModel(U, V, Sigma_U, Sigma_V, gammas)  # create new LR model
        E_YminusUV = Y - U @ V.T  # (m, n)
        tempmat = tempU.reshape(r*r, m).T @ tempV.reshape(r*r, n)
        E_YminusUV2 = Y**2 - 2 * Y * (U @ V.T) + tempmat
        return (
            lr_model,
            r,
            E_YminusUV,
            E_YminusUV2
        )

    def __mog_vmax(self, mog_model: MOGModel, mog_prior: MOGPrior, E_YminusUV: np.ndarray, E_YminusUV2: np.ndarray):
        alpha0 = mog_prior.alpha0
        beta0 = mog_prior.beta0
        mu0 = mog_prior.mu0
        c0 = mog_prior.c0
        d0 = mog_prior.d0
        R = mog_model.R

        m, n = E_YminusUV.shape
        k = R.shape[2]

        nxbar = R.reshape(m*n, k) @ E_YminusUV.reshape(-1)  # (k, )
        nk = np.sum(R.reshape(m*n, k), axis = 0)
        alpha = alpha0 + nk
        beta = beta0 + nk
        c = c0 + nk / 2
        mu = (beta0 * mu0 + nxbar) / beta
        d = d0 + 0.5 * (R.reshape(m*n, k) @ E_YminusUV2.reshape(-1) + beta0 * mu0**2 - (nxbar + beta0 * mu0)**2 / beta)

        mog_model = MOGModel(R, c, beta, d, mu)  # create new mog model
        return mog_model

    def __mog_vexp(self, mog_model: MOGModel, E_YminusUV: np.ndarray, E_YminusUV2: np.ndarray):
        alpha = mog_model.alpha
        beta = mog_model.beta
        mu = mog_model.mu
        c = mog_model.c
        d = mog_model.d

        m, n = E_YminusUV.shape
        k = mu.shape[0]
        Ex = E_YminusUV.reshape(-1)
        Ex2 = E_YminusUV2.reshape(-1)

        tau = c / d
        EQ = np.zeros(m * n, k)
        for i in range(k):
            EQ[:, i] = 1/beta[i] + tau[i] * mu[i]**2 + tau[i] * Ex2 - 2 * tau[i] * Ex * mu[i]

        Elogtau = psi(c) - np.log(d)
        Elogpi = psi(alpha) - psi(np.sum(alpha))

        logRho = (EQ - 2 * Elogpi+Elogtau - np.log(2*np.pi)) / (-2)
        logR = logRho - logsumexp(logRho, axis=1)
        R = np.exp(logR)

        mog_model.R = R.reshape(m, n, k)
        return mog_model

    def decompose(self, M: np.ndarray, rank =  None, lr_prior: LRPrior = None, mog_prior: MOGPrior = None):
        check_real_matrix(M)
        Y = M.copy()   # make a copy of the matrix to avoid side effects

        m, n = Y.shape
        mn = m * n
        k = self.mog_k

        # initialization
        if rank is None:
            rank = min(m, n)

        if lr_prior is None:
            lr_prior = LRPrior()

        if mog_prior is None:
            mog_prior = MOGPrior()

        # low rank model initialization
        Y2sum = np.sum(Y**2)
        scale2 = Y2sum / mn
        scale = np.sqrt(scale2)
        if self.init_method == "SVD":
            # SVD initialization
            u, s, vt = np.linalg.svd(Y, full_matrices=False)
            r = rank
            U = u[:, :r] @ np.diag(s[:r])
            Vt = np.diag(s[:r]) @ vt[:r, :]
            V = Vt.T
        else:
            # random initialization
            r = rank
            U = np.random.randn(m, r) * scale**0.5
            V = np.random.randn(n, r) * scale**0.5
        Sigma_U = np.tile(scale * np.eye(r), (1, 1, m) )
        Sigma_V = np.tile(scale * np.eye(r), (1, 1, n) )
        gammas = scale * np.ones(r)

        lr_model = LRModel(U, V, Sigma_U, Sigma_V, gammas)
        L = U @ V.T

        # Mog model initialization
        E = Y - L  # (m, n)
        mog_model = MOGModel(None, None, None, None, None)  # initialize blank mog model
        R = self.__R_initialization(E.reshape(-1), k)   # (mn, k)
        nk = np.sum(R, axis = 0)   # (k,)
        nxbar = R @ E.reshape(-1)  # (k, )
        mog_model.c = mog_prior.c0 + nk/2   # (k, )
        mog_model.beta = mog_prior.beta0 + nk   # (k, )
        mog_model.d = mog_prior.d0 + 0.5 * (R @ (E.reshape(-1) ** 2)) + mog_prior.beta0 * mog_prior.mu0**2 - 1/mog_model.beta * (nxbar + mog_prior.beta0 * mog_prior.mu0)**2   # (k, )
        mog_model.R = R.reshape(m, n, k)
        mog_model.mu = 1/mog_model.beta * (mog_prior.beta0 * mog_prior.mu0 + nxbar)   # (k, )

        # main loop
        for niter in range(self.maxiter):
            L_old = L

            # LR update
            lr_model, r, E_minusUV, E_minusUV2 = self.__lr_update(Y, lr_model, mog_model, r, lr_prior)
            L = lr_model.U @ lr_model.V.T

            # mog update
            mog_model = self.__mog_vmax(mog_model, mog_prior, E_minusUV, E_minusUV2)
            mog_model = self.__mog_vexp(mog_model, E_minusUV, E_minusUV2)
            

            # convergence check
            if np.linalg.norm(L - L_old, 'fro') / np.linalg.norm(L_old, 'fro') < self.tol:
                break

        return LSNResult(
            L = L,
            S = (Y - L),
            N = None,
            convergence = {
                'converged': niter < self.maxiter,
                'niter': niter,
            },
            lr_model = lr_model,    # additional parameters
            mog_model = mog_model,
        )







