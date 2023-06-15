import numpy as np

from ...utils.validations import check_real_matrix
from ...utils.constants import EPS
from ...base import RankFactorizationResult


class VariationalBayes:

    """
    Reference:
        S. D. Babacan, M. Luessi, R. Molina, and A. K. Katsaggelos, 
        "Sparse Bayesian Methods for Low-Rank Matrix Estimation," 
        IEEE Transactions on Signal Processing, 2012.
    """

    def __init__(self, **kwargs) -> None:
        self.verbose = kwargs.get("verbose", False)   # output the progress
        
        # initialization method: 
        #   'rand': initialize A and B with random matrices
        #   'ml': Apply SVD to Y and initialize A and B using its factors.
        self.initmethod = kwargs.get("init", "ml")
        assert self.initmethod in ["ml", "rand"], "Initialization method must be either 'rand' or 'ml'"

        self.a_gamma0 = kwargs.get("a_gamma0", 1e-6)
        self.b_gamma0 = kwargs.get("b_gamma0", 1e-6)

        self.a_alpha0 = kwargs.get("a_alpha0", 0)
        self.b_alpha0 = kwargs.get("b_alpha0", 0)

        self.a_beta0 = kwargs.get("a_beta0", 0)
        self.b_beta0 = kwargs.get("b_beta0", 0)

        # Flag for inference of component of E. 
        #   "standard"
        #   "fixed point"
        self.inference_flag = kwargs.get("inference_flag", "standard")
        assert self.inference_flag in ["standard", "fixed point"], "Inference flag must be either 'standard' or 'fixed point'"

        self.maxiter = kwargs.get("maxiter", 100)
        self.update_beta = kwargs.get("update_beta", True) # whether to update noise variance
        self.beta = kwargs.get("beta", None)
        self.dim_red = kwargs.get("dim_red", True)  # whether to prune irrelevant dimensions during the iterations
        self.dim_red_thr = kwargs.get("dim_red_thr", 1e4)   # the threshold to use to prune irrelevant dimensions
        self.mode = kwargs.get("mode", "VB")   # there are 3 modes, VB, VB_app, and MAP
        assert self.mode in ["VB", "VB_app", "MAP"], "Mode must be one of following: 'VB', 'VB_app' and 'MAP'"

        self.tol = kwargs.get("tol", 1e-5)  # the tolerance level for convergence

        
    def decompose(self, M: np.ndarray, rank: int = None):
        check_real_matrix(M)
        Y = np.copy(M)
        m, n = Y.shape
        mn = m * n
        Y2sum = np.sum(Y**2)
        scale2 = Y2sum / mn
        scale = np.sqrt(scale2)

        # Initialize A, B, and E
        if self.initmethod == "ml":
            U, s, Vh = np.linalg.svd(Y, full_matrices=False)
            V = Vh.T
            r = min(m, n) if rank is None else rank
            A = U[:, :r] @ np.diag(s[:r]**0.5)
            B = np.diag(s[:r]**0.5) @ V[:r, :].T
            
            Sigma_A = scale * np.eye(r)
            Sigma_B = scale * np.eye(r)
            gammas = scale * np.ones(r)
            beta = 1 / scale2

            if not self.update_beta and self.beta is not None:
                beta = self.beta
            
            Sigma_E = scale * np.ones((r, r))
            alphas = np.ones((m, n)) * scale
        elif self.initmethod == "rand":
            # random initialization
            r = min(m, n)
            A = np.random.randn(m, r) * np.sqrt(scale)
            B = np.random.randn(n, r) * np.sqrt(scale)
            gammas = scale * np.ones(r)
            Sigma_A = scale * np.eye(r)
            Sigma_B = scale * np.eye(r)
            E = np.random.randn(m, n) * np.sqrt(scale)
            beta = 1 / scale2
            Sigma_E = scale * np.ones((r, r))
            alphas = np.ones((m, n)) * scale
        else:
            raise ValueError("Invalid initialization method")
    
        X = A @ B.T
        E = Y - X
        metrics = {
            'niter': 0,
            'Econv': [],
            'Xconv': [],
            'converged': True
        }

        # Iterations
        for it in range(1, self.maxiter + 1):
            old_X = X
            old_E = E

            # Update X
            W = np.diag(gammas)

            # A step
            if self.mode == "VB":
                Sigma_A = np.linalg.inv(beta * B.T @ B + W + beta * n * Sigma_B)
                A = beta * (Y - E) @ B @ Sigma_A
            elif self.mode == "VB_app":
                Sigma_A = beta * B.T @ B + W + beta * n * Sigma_B
                A = np.linalg.solve(Sigma_A, beta * (Y - E) @ B)
                Sigma_A = np.diag(1/np.diag(Sigma_A))
            elif self.mode == "MAP":
                Sigma_A = beta * B.T @ B + W + beta * n * Sigma_B
                A = np.linalg.solve(Sigma_A, beta * (Y - E) @ B)
                Sigma_A = np.zeros_like(Sigma_A)

            # B step
            if self.mode == "VB":
                Sigma_B = np.linalg.inv(beta * A.T @ A + W + beta * m * Sigma_A)
                B = beta * (Y - E).T @ A @ Sigma_B
            elif self.mode == "VB_app":
                Sigma_B = beta * A.T @ A + W + beta * m * Sigma_A
                B = np.linalg.solve(Sigma_B, beta * (Y - E).T @ A)
                Sigma_B = np.diag(1/np.diag(Sigma_B))
            elif self.mode == "MAP":
                Sigma_B = beta * A.T @ A + W + beta * m * Sigma_A
                B = np.linalg.solve(Sigma_B, beta * (Y - E).T @ A)
                Sigma_B = np.zeros_like(Sigma_B)

            X = A @ B.T

            # E Step
            Sigma_E = 1/(alphas + beta)
            E = beta * (Y - X) * Sigma_E

            # Estimate alphas
            if self.inference_flag == "standard":
                alphas = 1 / (E**2 + Sigma_E)
            elif self.inference_flag == "fixed point":
                # MacKay fixed point method
                alphas = (1 - alphas * Sigma_E + self.a_alpha0 ) / (E**2 + EPS + self.b_alpha0)

            # Estimate gammas
            if self.inference_flag == "standard":
                gammas = (m + n + self.a_gamma0) / (np.diag(B.T @ B) + np.diag(A.T @ A) + m * np.diag(Sigma_A) + n * np.diag(Sigma_B) + self.b_gamma0)
            elif self.inference_flag == "fixed point":
                # MacKay fixed point method
                gammas = ( m + n - m * np.diag(Sigma_A) - n*np.diag(Sigma_B) + self.a_gamma0) / (np.diag(B.T @ B) + np.diag(A.T @ A) + self.b_gamma0)
            
            # Estimate beta
            if self.update_beta:
                err = np.sum((Y - X - E)**2) + n * np.trace(A.T @ A @ Sigma_B) + m * np.trace(B.T @ B @ Sigma_A) + mn * np.trace(Sigma_A @ Sigma_B) + np.sum(Sigma_E)
                beta = (mn + self.a_beta0) / (err + self.b_beta0)

            # Prune irrelevant dimension
            if self.dim_red:
                max_gamma = np.min(gammas) * self.dim_red_thr

                indices = gammas < max_gamma
                if np.sum(indices) > 0:
                    A = A[:, indices]
                    B = B[:, indices]
                    gammas = gammas[indices]

                    Sigma_A = Sigma_A[:, indices][indices, :]
                    Sigma_B = Sigma_B[:, indices][indices, :]

                    m, r = A.shape
                    n, r = B.shape
            
            # Check for convergence and display progress
            Xconv = np.linalg.norm(old_X - X, 'fro') / np.linalg.norm(old_X, 'fro')

            if self.verbose:
                Econv = np.linalg.norm(old_E - E, 'fro') / np.linalg.norm(old_E, 'fro')
                metrics['Econv'].append(Econv)
                metrics['Xconv'].append(Xconv)

            if it > 5 and Xconv < self.tol:
                break

        metrics['niter'] = it
        metrics['converged'] = (it < self.maxiter)   # whether maxit reach or converged

        # return result
        if self.verbose:
            return RankFactorizationResult(
                A = A,
                B = B,
                convergence = metrics,
                gamma = gammas,
                alpha = alphas,
                beta = betas,
                Sigma_A = Sigma_A,
                Sigma_B = Sigma_B
            )
        else:
            return RankFactorizationResult(
                A = A,
                B = B,
                convergence = metrics
            )
                

        

