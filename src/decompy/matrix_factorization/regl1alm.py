import numpy as np
from typing import Union

from ..utils.validations import check_real_matrix, check_binary_matrix
from ..interfaces import RankFactorizationResult


class RegulaizedL1AugmentedLagrangianMethod:
    """
    Implements the Regularized L1 Augmented Lagrangian algorithm for low rank matrix factorization
    with trace norm regularization.

    Robust low-rank matrix approximation with missing data and outliers
        min |W.*(M-E)|_1 + lambda*|V|_*
        s.t., E = UV, U'*U = I

    Notes
    -----
    [1] Y. Zheng, G. Liu, S. Sugimoto, S. Yan and M. Okutomi, "Practical low-rank matrix approximation under robust L1-norm," 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, 2012, pp. 1410-1417, doi: 10.1109/CVPR.2012.6247828. keywords: {Robustness;Optimization;Convergence;Computer vision;Approximation algorithms;Least squares approximation},
    """

    def __init__(self, **kwargs):
        """Initialize the Regularized L1 Augmented Lagrangian Method class.

        Parameters
        ----------
        maxiter_in : int, optional
            Maximum number of inner iterations. Default is 100.
        maxiter_out : int, optional
            Maximum number of outer iterations. Default is 5000.
        rho : float, optional
            Penalty parameter. Default is 1.05.
        max_mu : float, optional
            Maximum value for penalty parameter. Default is 1e20.
        tol : float, optional
            Tolerance for stopping criteria. Default is 1e-8.

        """
        self.maxiter_in = kwargs.get("maxiter_in", 100)
        self.maxiter_out = kwargs.get("maxiter_out", 5000)
        self.rho = kwargs.get("rho", 1.05)
        self.max_mu = kwargs.get("max_mu", 1e20)
        self.tol = kwargs.get("tol", 1e-8)

    def decompose(
        self,
        D: np.ndarray,
        W: Union[np.ndarray, None] = None,
        r=None,
        lambd: Union[float, None] = None,
    ):
        """Decompose a matrix D into low rank factors U and V.

        Parameters
        ----------
        D : ndarray
            The m x n data matrix to decompose.
        W : ndarray or None, optional
            The m x n indicator matrix, with 1 representing observed entries
            and 0 representing missing entries. Default is None, which means
            all entries are observed.
        r : int or None, optional
            The rank of the decomposition. If None, default is ceil(0.1*min(m,n)).
        lambd : float or None, optional
            The regularization parameter. Default is 1e-3.

        Returns
        -------
        res : RankFactorizationResult
            A named tuple containing the low rank factors U and V, and convergence
            info such as number of iterations, error, etc.

        """
        check_real_matrix(D)
        M = D.copy()  # create a copy of the matrix
        m, n = M.shape
        if W is None:
            W = np.ones_like(D)
        check_binary_matrix(W)
        if r is None:
            r = np.ceil(0.1 * min(m, n))
        if lambd is None:
            lambd = 1e-3

        # normalization
        scale = np.max(np.abs(M))
        M /= scale

        # initialization
        mu = 1e-6
        M_norm = np.linalg.norm(M, "fro")
        tol = self.tol * M_norm

        cW = np.ones_like(W) - W  # the complement of W
        E = np.zeros((m, n))
        U = np.zeros((m, r))
        V = np.zeros((r, n))
        Y = np.zeros((m, n))  # lagrange multiplier

        # start main outer loop
        niter_out = 0
        while niter_out < self.maxiter_out:
            niter_out += 1

            niter_in = 0
            obj_pre = 1e20

            while niter_in < self.maxiter_in:
                # update U
                temp = (E + Y / mu) @ V.T
                Us, sigma, Udt = np.linalg.svd(temp, full_matrices=False)  # stable
                U = Us @ Udt

                # update V
                temp = U.T @ (E + Y / mu)
                Vs, sigma, Vdt = np.linalg.svd(temp, full_matrices=False)  # stable
                svp = np.sum(sigma > lambd / mu)
                if svp >= 1:
                    sigma = sigma[:svp] - lambd / mu
                else:
                    svp = 1
                    sigma = np.array([0])
                V = Vs[:, :svp] @ np.diag(sigma) @ Vdt[:svp, :]
                sigma0 = sigma

                UV = U @ V

                # update E
                temp1 = UV - Y / mu
                temp = M - temp1
                E = np.maximum(temp - 1 / mu, 0) + np.minimum(temp + 1 / mu, 0)
                E = (M - E) * W + temp1 * cW

                # evaluate current objective
                obj_cur = (
                    np.sum(np.abs(W * (M - E)))
                    + lambd * np.sum(sigma0)
                    + np.sum(np.abs(Y * (E - UV)))
                    + mu / 2 * np.linalg.norm(E - UV, "fro") ** 2
                )

                # check convergence of inner loop
                if np.abs(obj_cur - obj_pre) < 1e-8 * np.abs(obj_pre):
                    break
                else:
                    obj_pre = obj_cur
                    niter_in += 1

            leq = E - UV
            stop_c = np.linalg.norm(leq, "fro")
            if stop_c < tol:
                break
            else:
                # update lagrange multiplier
                Y += mu * leq
                mu = min(mu * self.rho, self.max_mu)  # update penalty parameter

        # denormalization
        U_est = np.sqrt(scale) * U
        V_est = np.sqrt(scale) * V
        M_est = U_est @ V_est
        l1_error = np.sum(np.abs(W * (scale * M - M_est)))

        return RankFactorizationResult(
            A=U_est,
            B=V_est.T,
            convergence={
                "niter": niter_out,
                "stop_c": stop_c,
                "l1_error": l1_error,
                "converged": (niter_out < self.maxiter_out),
            },
        )
