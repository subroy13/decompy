import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import LSNResult


class L1Filtering:
    """
    Robust PCA using L1 Filtering

    Notes
    ------
    [1] Liu, R., Lin, Z., Wei, S., & Su, Z. (2011). Solving principal component pursuit in linear time via $ l_1 $ filtering. arXiv preprint arXiv:1108.5359.
    """

    def __init__(self, **kwargs):
        """Initialize the L1F class.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Default is 1e6.
        tol : float, optional
            Tolerance for stopping criteria. Default is 1e-7.
        rho : float, optional
            Update parameter for augmented Lagrangian. Default is 1.1.
        max_mu : float, optional
            Maximum value for mu. Default is 1e30.

        """
        self.maxiter = kwargs.get("maxiter", 1e6)
        self.tol = kwargs.get("tol", 1e-7)
        self.rho = kwargs.get("rho", 1.1)
        self.max_mu = kwargs.get("max_mu", 1e30)

    def _generate_seed(self, D: np.ndarray, sr, sc):
        """Generate seeds i.e., initialization values for the L1 filtering

        Parameters
        ----------
        D : numpy.ndarray
            The input matrix
        sr : int
            Desired number of rows in seed matrix
        sc : int
            Desired number of columns in seed matrix

        Returns
        -------
        A_seed : numpy.ndarray
            Seeded low-rank approximation of D
        column_seed : numpy.ndarray
            Column indices for seed matrix
        row_seed : numpy.ndarray
            Row indices for seed matrix
        rA : int
            Rank of the seed matrix

        Generates a low rank seed matrix for the input matrix D by randomly
        sampling rows and columns. The number of rows and columns sampled
        is determined by the desired size sr and sc. Singular value
        decomposition is used to find the approximate rank rA of the
        sampled matrix, and more rows/columns are sampled if needed to
        achieve the desired size sr * rA and sc * rA. The seeded low rank
        approximation A_seed is returned along with the sampled row and column
        indices.
        """
        if sr is None:
            sr = 10
        if sc is None:
            sc = 10
        thres = 1e-3

        m, n = D.shape

        dr = np.ceil(m / sr).astype('int')
        dc = np.ceil(n / sc).astype('int')

        row_seed = np.random.permutation(m)[:dr]
        column_seed = np.random.permutation(n)[:dc]

        D_seed = D[row_seed, :][:, column_seed]
        _, s, _ = np.linalg.svd(D_seed, full_matrices=False)
        rA = np.sum(s > thres * s[0])  # get the approximate low rank

        # estimate the rank
        while dr / rA < sr or dc / rA < sc or max(m / dr, n / dc) < 0.5:
            dr, dc = sr * rA, sc * rA

            # take another random permutation
            row_seed = np.random.permutation(m)[:dr]
            column_seed = np.random.permutation(n)[:dc]

            D_seed = D[row_seed, :][:,column_seed]
            _, s, _ = np.linalg.svd(D_seed, full_matrices=False)
            rA = np.sum(s > thres * s[0])  # get the approximate low rank

        true_r = rA

        # seed recovery
        dr = sr * true_r
        dc = sc * true_r
        row_seed = np.random.permutation(m)[:dr]
        column_seed = np.random.permutation(n)[:dc]

        D_seed = D[row_seed, :][:, column_seed]
        U, s, Vt = np.linalg.svd(D_seed, full_matrices=False)
        rA = np.sum(s > thres * s[0])  # get the approximate low rank
        A_seed = U[:, :rA] @ np.diag(s[:rA]) @ Vt[:rA, :]

        return (A_seed, column_seed, row_seed, rA)

    def _solve_ml1(self, D: np.ndarray, A: np.ndarray, pinvA: np.ndarray):
        """Solve the L1-norm minimization problem.

        Parameters
        ----------
        D : ndarray
            The n x s matrix D.
        A : ndarray
            The n x m matrix A.
        pinvA : ndarray
            The pseudo-inverse of A. If None, it will be computed.

        Returns
        -------
        E : ndarray
            The n x s matrix E.
        alpha : ndarray
            The m x s matrix alpha.
        niter : int
            Number of iterations performed.
        stop_c : float
            The maximum absolute value of the residual.

        Solves the optimization problem:
            min_{E,alpha} |E|_1
            s.t A*alpha+E=D
        Where E, D are n x s matrices,
        A is a n x m matrix, and alpha is a m x s matrix.
        """
        n, m = A.shape
        s = D.shape[1]

        if pinvA is None:
            pinvA = np.linalg.pinv(A)

        # initialization
        alpha = np.zeros((m, s))
        E = np.zeros((n, s))
        Y = np.zeros((n, s))
        mu = 1e-6

        # start main loop
        niter = 0
        while niter < self.maxiter:
            niter += 1

            # update E
            temp_T = D - A @ alpha + (1 / mu) * Y   # (n, s)
            E = np.maximum(temp_T - 1 / mu, 0) + np.minimum(temp_T + 1 / mu, 0)

            # update alpha
            temp_T = D + (1 / mu) * Y - E
            alpha = pinvA @ temp_T

            leq = D - A @ alpha - E
            stop_c = (np.abs(leq)).max()

            if stop_c < self.tol:
                break
            else:
                # update Y
                Y += mu * leq

                # update mu
                mu = min(self.max_mu, mu * self.rho)

        return (E, alpha, niter, stop_c)

    def decompose(self, M: np.ndarray, sr=1, sc=1):
        """Decompose a matrix M into low-rank, sparse and noise components using L1 factorization.

        Parameters
        ----------
        M : ndarray
            Input matrix to decompose.
        sr : int, optional
            Number of rows to sample for initialization. Default is 1.
        sc : int, optional
            Number of columns to sample for initialization. Default is same as sr.

        Returns
        -------
        LSNResult
            A named tuple containing the low-rank, sparse and noise components.
            L : ndarray
                Low-rank component
            S : ndarray
                Sparse component
            N : ndarray
                Noise component (zeros)
            convergence : dict
                Dictionary containing convergence info with keys:
                    'iteration': number of iterations
                    'converged': boolean indicating if converged
                    'error': final error value

        """
        if sr is None:
            sr = 1
        if sc is None:
            sc = sr

        check_real_matrix(M)
        D = (
            M.copy()
        )  # make a copy of the matrix so that original matrix remains unchanged
        m, n = D.shape

        # applies SVD on a partial matrix for initialization
        (A_seed, column_seed, row_seed, rA) = self._generate_seed(D, sr, sc)

        U, S, Vt = np.linalg.svd(A_seed, full_matrices=False)
        Ur, Sr, Vtr = U[:, :rA], S[:rA], Vt[:rA, :]

        AS_row = Ur @ np.diag(Sr)
        AS_column = np.diag(Sr) @ Vtr

        A_row = np.zeros((rA, n))
        A_column = np.zeros((rA, m))

        column_comp = [i for i in range(n) if i not in column_seed.tolist()]
        _, Ac_row, niterrow, stopc_row = self._solve_ml1(
            D[row_seed, :][:, column_comp], Ur, Ur.T
        )
        A_row[:, np.array(column_comp)] = Ac_row
        A_row[:, column_seed] = AS_row

        row_comp = [i for i in range(m) if i not in row_seed.tolist()]
        _, Ac_column, nitercol, stopc_col = self._solve_ml1(
            D[row_comp, :][:, column_seed].T, Vtr.T, Vtr
        )
        A_column[:, np.array(row_comp)] = Ac_column
        A_column[:, row_seed] = AS_column.T

        A_full = A_column.T @ np.diag(1 / Sr) @ A_row
        E_full = D - A_full

        return LSNResult(
            L=A_full,
            S=E_full,
            N=None,
            convergence={
                "iteration": max(niterrow, nitercol),
                "converged": (niterrow < self.maxiter) and (nitercol < self.maxiter),
                "error": max(stopc_row, stopc_col),
            },
        )
