import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import PCAResult


class MEstimation:
    """
    Robust PCA using Classical M-estimation method

    Notes
    -----
    [1] F. De la Torre and M. J. Black, "Robust principal component analysis for computer vision," Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, Vancouver, BC, Canada, 2001, pp. 362-369 vol.1, doi: 10.1109/ICCV.2001.937541.

    """

    def __init__(self, **kwargs):
        """Initialize the M-Estimator class.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Default is 300.
        verbose : bool, optional
            Whether to print optimization progress. Default is False.
        iter_grad : int, optional
            Gradient is estimated every `iter_grad` iterations. Default is 2.
        mu : float, optional
            Regularization parameter. Default is 1.
        """
        self.maxiter = kwargs.get("maxiter", 300)
        self.verbose = kwargs.get("verbose", False)
        self.iter_grad = kwargs.get("iter_grad", 2)
        self.mu = kwargs.get("mu", 1)

    def _compute_scale_statistics(self, M: np.ndarray, ini_rank: int = None):
        """Compute scale statistics from input matrix M.

        Parameters
        ----------
        M : ndarray
            Input matrix.
        ini_rank : int, optional
            Initial rank for SVD computation. If None, set to ceil(M.shape[0]/10).

        Returns
        -------
        Sigmai : ndarray
            Scale matrix for input.
        Sigmaf : ndarray
            Scale matrix for error.
        basis_ini : ndarray
            Initial basis matrix.

        """
        # compute standard PCA
        mean_ls = np.mean(M, axis=0)
        U, _, VT = np.linalg.svd(M - mean_ls, full_matrices=False)
        if ini_rank is None:
            ini_rank = np.ceil(M.shape[0] / 10)
        U = U[:, :ini_rank]
        VT = VT[:ini_rank, :]

        sizeim = M.shape
        Sigmaf = np.zeros(sizeim)
        cini = U.T @ (M - mean_ls)
        error = M - mean_ls - U @ cini
        medianat = np.median(np.abs(error))
        error2 = error - medianat
        Sigmaft = np.sqrt(3) * 1.4826 * np.median(np.abs(error2))
        Nl = 0
        for i in range(Nl + 1, sizeim[0] - Nl - 1):
            for j in range(Nl + 1, sizeim[1] - Nl - 1):
                y, x = np.meshgrid(range(i - Nl, i + Nl + 1), range(j - Nl, j + Nl + 1))
                ind = np.ravel_multi_index((y, x), sizeim)
                errorlo = error[ind, :]
                medianat = np.median(np.abs(errorlo))
                error2 = error[ind, :] - medianat
                Sigmaf[i, j] = 2.3 * np.sqrt(3) * 1.4826 * np.median(np.abs(error2))
        Sigmaf = np.maximum(Sigmaf, Sigmaft * np.ones(Sigmaf.shape))
        Sigmai = 3 * Sigmaf

        basis_ini = U + np.random.randn(M.shape[0], ini_rank)
        return Sigmai, Sigmaf, basis_ini

    def decompose(self, M: np.ndarray, rank: int = None):
        """Perform robust M-Estimation based matrix factorization on input matrix M.

        Parameters
        ----------
        M : ndarray
            Input matrix to decompose, must be a real matrix.
        rank : int, optional
            Rank of decomposition. If not provided, rank is set to
            min(M.shape).

        Returns
        -------
        pca_result : PCAResult
            Results of MEST factorization containing components.

        Examples
        --------
        >>> import numpy as np
        >>> from decompy.matrix_factorization import MEstimation
        >>> X = np.random.rand(100,200)
        >>> mest = MEstimation()
        >>> results = mest.decompose(X)
        """
        check_real_matrix(M)
        X = M.copy()

        # Do initialization
        info = np.zeros((self.maxiter, 2))
        Sigma_end, Sigma_start, Bg = self._compute_scale_statistics(X, rank)
        rob_mean = np.median(X, axis=0)
        cg = np.linalg.pinv(Bg) @ (X - rob_mean)
        Sigma = Sigma_start

        niter = 1
        converged = False
        errsvd = 0

        # main iteration loop
        while niter < self.maxiter and not converged:
            sigma_max_ant = np.max(Sigma)
            sigma_min_ant = np.min(Sigma)
            Sigma *= 0.92
            Sigma = np.maximum(Sigma, Sigma_end)
            Sigmat = np.square(Sigma) @ np.ones((1, X.shape[1]))

            Bgant = Bg

            # updating the mean
            for _ in range(self.iter_grad):
                error = X - rob_mean - Bg @ cg
                pondera = error * Sigmat / ((Sigmat + error**2) ** 2)
                rob_mean += (
                    self.mu * np.sum(pondera, axis=0) / (X.shape[1] / Sigmat[:, 0])
                )

            # Computing the error
            errsvdant = errsvd
            errsvd = np.sum(((error**2) / (Sigmat + error**2)))

            # updating the basis
            for _ in range(self.iter_grad):
                error = X - rob_mean - Bg @ cg
                pondera = error * Sigmat / ((Sigmat + error**2) ** 2)
                Bg += (self.mu * pondera @ cg.T) / ((1 / Sigmat) @ (cg**2).T)

            # updating the coefficients
            for _ in range(self.iter_grad):
                error = X - rob_mean - Bg @ cg
                pondera = error * Sigmat / ((Sigmat + error**2) ** 2)
                cg += (self.mu * Bg.T @ pondera) / ((Bg**2).T @ (1 / Sigmat))

            # angular error
            angular_error = np.rad2deg(
                np.arccos(np.clip(np.trace(Bg.T @ Bgant), -1, 1))
            )

            if self.verbose:
                print(
                    f"Iter: {niter}, Err: {errsvd:.3f}, Sigmax: {np.max(Sigma):.4e}, Sigmin: {np.min(Sigma):.4e}, Angular error: {angular_error:.4f}, mu: {self.mu:.2f}"
                )

            info[niter - 1, :] = [errsvd, angular_error]

            # convergence check
            if (
                errsvd > errsvdant
                and sigma_max_ant == np.max(Sigma)
                and sigma_min_ant == np.min(Sigma)
            ):
                self.mu *= 0.9

            if angular_error < 1e-4 and niter > 30:
                # do at least 30 iterations
                converged = True

            niter += 1

        return PCAResult(
            loc=rob_mean,
            eval=cg,
            evec=Bg,
            convergence={
                "info": info,
                "niter": niter,
                "converged": (niter < self.maxiter),
            },
        )
