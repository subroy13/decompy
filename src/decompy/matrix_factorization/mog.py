import numpy as np

from ..utils.validations import check_real_matrix

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


    def decompose(self, M: np.ndarray, rank =  None, lr_prior = None, mog_prior = None):
        check_real_matrix(M)
        Y = M.copy()   # make a copy of the matrix to avoid side effects

        m, n = Y.shape
        mn = m * n
        k = self.mog_k

        # initialization
        if rank is None:
            rank = min(m, n)

        if lr_prior is None:
            lr_prior = {}
        a0 = lr_prior.get("a0", 1e-6)
        b0 = lr_prior.get("b0", 1e-6)

        if mog_prior is None:
            mog_prior = {}
        mu0 = mog_prior.get("mu0", 0)
        c0 = mog_prior.get("c0", 1e-6)
        d0 = mog_prior.get("d0", 1e-6)
        alpha0 = mog_prior.get("alpha0", 1e-6)
        beta0 = mog_prior.get("beta0", 1e-6)

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

        L = U @ V.T

        # Mog model initialization
        E = Y - L  # (m, n)
        R = self.__R_initialization(E.reshape(-1), k)   # (mn, k)
        nk = np.sum(R, axis = 0)   # (k,)
        nxbar = R @ E.reshape(-1)  # (k, )
        c = c0 + nk/2   # (k, )
        beta = beta0 + nk   # (k, )
        d = d0 + 0.5 * (R @ (E.reshape(-1) ** 2)) + beta0 * mu0**2 - 1/beta * (nxbar + beta0 * mu0)**2   # (k, )
        R = R.reshape(m, n, k)
        mu = 1/beta * (beta0 * mu0 + nxbar)   # (k, )

        # main loop
        for niter in range(self.maxiter):
            L_old = L

            # LR update
            








