import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult


class ExactAugmentedLagrangianMethod:
    """
    Robust PCA using Exact Augmented Lagrangian Multiplier Method

    Notes
    -----
    [1] Lin, Zhouchen, Minming Chen, and Yi Ma. "The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices." arXiv preprint arXiv:1009.5055 (2010).

    """

    def __init__(self, **kwargs):
        """Initialize Exact Augmented Lagrangian Method solver.

        Parameters
        ----------
        tol : float, optional
            Tolerance for stopping criteria.
        maxit : int, optional
            Maximum number of iterations.
        maxit_inner : int, optional
            Maximum number of inner iterations.
        verbose : bool, optional
            Print status messages if True.

        """
        self.tol = kwargs.get("tol", 1e-7)
        self.maxit = kwargs.get("maxit", 1e3)
        self.maxit_inner = kwargs.get("maxit_inner", 100)
        self.verbose = kwargs.get("verbose", False)

    def decompose(
        self, M: np.ndarray, lambd: float = None, mu: float = None, rho: float = 6
    ):
        """Decompose a matrix M into a low-rank matrix L and a sparse matrix S.

        Parameters
        ----------
        M : ndarray
            Input matrix to decompose

        lambd : float, optional
            Weight on sparse error term in cost function.
            Default is 1/sqrt(m) where m is number of rows in M

        mu : float, optional
            Regularization parameter for low-rank matrix.
            Default is 0.5 / norm_two where norm_two is L2 norm of M

        rho : float, default 6
            Parameter for increasing mu at each iteration

        Returns
        -------
        LSNResult
            Named tuple containing low-rank matrix L, sparse matrix S
            and convergence info
        """

        check_real_matrix(M)
        D = M.copy()
        m, n = D.shape
        Y = np.sign(D)

        if lambd is None:
            lambd = 1 / np.sqrt(m)

        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y, np.Inf) / lambd
        dual_norm = max(norm_two, norm_inf)
        Y /= dual_norm

        A_hat = np.zeros_like(D)
        E_hat = np.zeros_like(D)
        dnorm = np.linalg.norm(D, "fro")
        tol_proj = 1e-6 * dnorm
        total_svd = 0
        if mu is None:
            mu = 0.5 / norm_two

        iter = 0
        converged = False
        sv = 5
        svp = sv
        while not converged:
            iter += 1

            # solve the primal problem by alternative projection
            primal_converged = False
            primal_iter = 0
            sv += np.round(n * 0.1)
            while not primal_converged:
                temp_T = D - A_hat + (1 / mu) * Y
                temp_E = np.maximum(temp_T - lambd / mu, 0) + np.minimum(
                    temp_T + lambd / mu, 0
                )
                U, diagS, VT = np.linalg.svd(
                    D - temp_E + (1 / mu) * Y, full_matrices=False
                )
                svp = (diagS > 1 / mu).sum()
                sv = min(svp + 1, n) if svp < sv else min(svp + np.round(0.05 * n), n)
                temp_A = U[:, :svp] @ np.diag(diagS[:svp] - 1 / mu) @ VT[:svp, :]

                if (
                    np.linalg.norm(A_hat - temp_A, "fro") < tol_proj
                    and np.linalg.norm(E_hat - temp_E, "fro") < tol_proj
                ):
                    primal_converged = True
                elif primal_iter > self.maxit_inner:
                    primal_converged = True

                A_hat, E_hat = temp_A, temp_E
                primal_iter += 1
                total_svd += 1

            Z = D - A_hat - E_hat
            Y += mu * Z
            mu *= rho

            # stopping criterion
            if np.linalg.norm(Z, "fro") / dnorm < self.tol:
                converged = True

            if self.verbose:
                print(
                    f"Iteration: {iter}, SVD: {total_svd}, Sparse Size: {(E_hat > 0).sum()}"
                )

            if not converged and iter >= self.maxit:
                converged = True

        return LSNResult(
            L=A_hat,
            S=E_hat,
            convergence={"iteration": iter, "converged": (iter < self.maxit)},
        )
