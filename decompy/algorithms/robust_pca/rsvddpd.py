from typing import Union
import numpy as np
import warnings

from ...utils.validations import check_real_matrix, is_decreasing
from ...base import SVDResult

class DensityPowerDivergence:

    def __init__(self, **kwargs) -> None:
        self.alpha = kwargs.get("alpha", 0.5)  # the alpha in the robust minimum divergence
        assert self.alpha >= 0 and self.alpha <= 1, "alpha must be between 0 and 1"   # * Protect against absurd options
        self.tol = kwargs.get("tol", 1e-4)     # * Minimum tolerance level for the norm of X
        self.eps = kwargs.get("eps", 1e-8)     # * Convergence criterion for change in left / right vectors
        self.maxiter = kwargs.get("maxiter", 1e2)  # * Maximum number of iterations per singular value
        self.inneriter = kwargs.get("inneriter", 5)   # * Number of inner fixed point iteration
        self.verbose = kwargs.get("verbose", False)

    def sanity_check(self, X: np.ndarray, lambdas: np.ndarray, maxit_reached: np.ndarray):
        """
            Does some simple sanity check on the output
        """
        # * Check if maxit is reached
        if maxit_reached.sum() > 0:
            warnings.warn("Maximum iteration reached for singular values " + ",".join([str(i) for i in np.where(maxit_reached)[0] + 1]), stacklevel=1)
        if not is_decreasing(lambdas):
            warnings.warn("Singular values are not in sorted order")


    def decompose(self, M: np.ndarray, rank: Union[int, None] = None, initu: Union[np.ndarray, None] = None, initv: Union[np.ndarray, None] = None):
        check_real_matrix(M)
        X = np.copy(M)   # create a copy of the matrix so you do not mutuate the existing one
        n, p = X.shape

        # Protect against absurd options
        if rank is not None:
            rank = max(1, min( rank, min(n, p) ))
        else:
            rank = min(n, p)

        # initialize initU, initV
        initu = initu if initu is not None else np.random.random((n, rank))
        initv = initv if initv is not None else np.random.random((p, rank))
        assert initu.shape[0] == n and initv.shape[0] == p and initu.shape[1] == rank and initv.shape[1] == rank, "Invalid initialization value"
        for i in range(rank):
            initu[:, i] = initu[:, i] / np.linalg.norm(initu[:, i])
            initv[:, i] = initv[:, i] / np.linalg.norm(initv[:, i])
        lambdas = np.zeros(rank)
        iterlist = np.zeros(rank)

        # normalize entries for better numerical precision
        scale_factor = np.quantile(X, [0.25, 0.75])
        X /= (scale_factor[1] - scale_factor[0])
        Xnorm = np.linalg.norm(X)

        for r in range(rank):
            curr_relnorm = np.linalg.norm(X) / Xnorm

            if curr_relnorm > self.tol:
                niter = 0
                converged = False
                sigma = 1

                while not converged:
                    curr_a = initu[:, r]
                    curr_b = initv[:, r]
                    curr_lambda = lambdas[r]

                    # STEP 1: Fix Right, Optimize Left
                    # Do fixed point iteration
                    left_iter = 0
                    fixed = False

                    c = np.zeros(n)

                    while not fixed:
                        curr_c = c
                        wmat = X - np.outer(c, curr_b)
                        wmat = np.exp(-0.5 * self.alpha * (wmat ** 2) / sigma)
                        numer = np.multiply(wmat * X, curr_b).sum(axis = 1)
                        denom = np.multiply(wmat, curr_b**2).sum(axis = 1)
                        c = numer / denom 
                        left_iter += 1

                        # Check if fixed point criterion is met
                        fixed = (np.linalg.norm(c - curr_c) / np.linalg.norm(c) < self.eps) or (left_iter > self.inneriter)

                    # Apply Gram-Schimdt Operation
                    if r > 0:
                        c -= ((initu[:, :r] @ initu[:, :r].T) @ c)
                    
                    # Normalized
                    curr_lambda = np.linalg.norm(c)
                    curr_a = c / curr_lambda

                    # STEP 2: Fix Left, Optimize Right
                    # Do fixed point iteration
                    right_iter = 0
                    fixed = False

                    d = np.zeros(p)

                    while not fixed:
                        curr_d = d
                        wmat = X - np.outer(curr_a, curr_d)
                        wmat = np.exp(-0.5 * self.alpha * (wmat ** 2) / sigma)
                        numer = np.multiply((wmat * X).T, curr_a).sum(axis = 1)
                        denom = np.multiply(wmat.T, curr_a**2).sum(axis = 1)
                        d = numer / denom 
                        right_iter += 1

                        # Check if fixed point criterion is met
                        fixed = (np.linalg.norm(d - curr_d) / np.linalg.norm(d) < self.eps) or (right_iter > self.inneriter)
                        
                    # * Apply Gram-Schimdt Criterion
                    if r > 0:
                        d -=((initv[:, :r] @ initv[:, :r].T) @ d)

                    # Normalize
                    curr_lambda = np.linalg.norm(d)
                    curr_b = d / curr_lambda

                    # STEP 3: Check if convergence is met
                    niter += 1
                    is_convl = (np.abs(curr_lambda - lambdas[r]) / (np.abs(lambdas[r]) + self.eps) < self.eps)
                    is_conva = np.linalg.norm(curr_a - initu[:, r])
                    is_convb = np.linalg.norm(curr_b - initv[:, r])

                    converged = (niter > self.maxiter) or (is_convl and is_conva and is_convb)
                    
                    lambdas[r] = curr_lambda
                    initu[:, r] = curr_a
                    initv[:, r] = curr_b

                # Outside iteration, count iteration
                iterlist[r] = niter

                # One singular value object, proceed to the next
                X -= (lambdas[r] * np.outer(initu[:, r], initv[:, r]))
            else:
                break
        
        # Change the singular values as required
        lambdas = (scale_factor[1] - scale_factor[0]) * lambdas
        initu = initu[:, :rank]    # Subset only the required column upto the rank
        initv = initv[:, :rank]    

        if self.verbose:
            self.sanity_check(M, lambdas, (iterlist >= self.maxiter))
        
        return SVDResult(initu, lambdas, initv, convergence = { 'iterations': iterlist })
