from typing import Union
import numpy as np
import warnings

from ..utils.validations import check_real_matrix, is_decreasing
from ..utils.constants import EPS
from ..base import SVDResult

class DensityPowerDivergence:

    def __init__(self, **kwargs) -> None:
        self.alpha = kwargs.get("alpha", 0.5)  # the alpha in the robust minimum divergence
        assert self.alpha >= 0 and self.alpha <= 1, "alpha must be between 0 and 1"   # * Protect against absurd options
        self.tol = kwargs.get("tol", 1e-4)     # * Minimum tolerance level for the norm of X
        self.eps = kwargs.get("eps", 1e-8)     # * Convergence criterion for change in left / right vectors
        self.maxiter = kwargs.get("maxiter", 1e2)  # * Maximum number of iterations per singular value
        self.inneriter = kwargs.get("inneriter", 5)   # * Number of inner fixed point iteration
        self.verbose = kwargs.get("verbose", False)
        self.method = kwargs.get("method", "v2")   # * The method of regression
        assert self.method in ["v1", "v2"] 
        self.initmethod = kwargs.get("initmethod", "random")   # the method of vector initialization
        assert self.initmethod in ["random", "tsvd"]


    def sanity_check(self, X: np.ndarray, lambdas: np.ndarray, maxit_reached: np.ndarray):
        """
            Does some simple sanity check on the output
        """
        # * Check if maxit is reached
        if maxit_reached.sum() > 0:
            warnings.warn("Maximum iteration reached for singular values " + ",".join([str(i) for i in np.where(maxit_reached)[0] + 1]), stacklevel=1)
        if not is_decreasing(lambdas):
            warnings.warn("Singular values are not in sorted order")

    def __decompose_v1(self, M: np.ndarray, rank: int, initu: np.ndarray, initv: np.ndarray):
        X = np.copy(M)
        n, p = X.shape
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
    
        return (lambdas, initu, initv, iterlist)


    def __decompose_v2(self, M: np.ndarray, rank: int, U: np.ndarray, V: np.ndarray):
        X = np.copy(M)
        n, p = X.shape

        # normalize entries for better numerical precision
        scale_factor = np.quantile(X, [0.25, 0.75])
        X /= (scale_factor[1] - scale_factor[0])

        niter = 0
        sigs = np.ones(rank)  # (r, )
        sigma = np.var(X) ** 0.5  # sigma

        while True:
            # step 1: estimate V based on U
            newsigs = sigs
            for _ in range(self.inneriter):
                ermat = X - U @ np.diag(newsigs) @ V.T   # (n, p)
                wmat = np.exp(-self.alpha * ermat**2 / (2 * (sigma **2) ))   # (n, p)
                for j in range(p):
                    V[j,:] = np.linalg.solve(U.T @ np.diag(wmat[:, j] + EPS) @ U, U.T @ np.diag(wmat[:, j]) @ X[:, j])   # (r, )

                # perform QR decomposition for gram schimdt
                V, R = np.linalg.qr(V)
                newsigs = np.absolute(np.diag(R))

            # step 2: estimate U based on V
            for _ in range(self.inneriter):
                ermat = X - U @ np.diag(newsigs) @ V.T   # (n, p)
                wmat = np.exp(-self.alpha * ermat**2 / (2 * (sigma **2) ))   # (n, p)
                for i in range(n):
                    U[i, :] = np.linalg.solve(V.T @ np.diag(wmat[i, :] + EPS) @ V, V.T @ np.diag(wmat[i, :]) @ X[i, :] )  # (r, )
                # perform QR decomposition for gram schimdt
                U, R = np.linalg.qr(U)
                newsigs = np.absolute(np.diag(R))

            # step 3: check convergence
            err = np.linalg.norm(sigs - newsigs) / np.linalg.norm(sigs)
            niter += 1
            if err < self.tol or niter >= self.maxiter:
                break
            else:
                sigs = newsigs
        
        # Change the singular values as required
        sigs = (scale_factor[1] - scale_factor[0]) * sigs
        return (sigs, U, V, niter)

    
    def get_initialized_U_V(self, M: np.ndarray, rank):
        X = np.copy(M)
        n, p = X.shape
        if self.initmethod == "random":
            initu = np.random.random((n, rank))
            initv = np.random.random((p, rank))
            
            # make it orthogonalized
            initu, _ = np.linalg.qr(initu)
            initv, _ = np.linalg.qr(initv)
        else:
            quants = np.quantile(X, q = [0.25, 0.75])
            X[X < quants[0]] = quants[0]
            X[X > quants[1]] = quants[1]
            U, _, Vt = np.linalg.svd(X)
            initu, initv = U, Vt.T
        return (initu, initv)



    def decompose(self, M: np.ndarray, rank: Union[int, None] = None, initu: Union[np.ndarray, None] = None, initv: Union[np.ndarray, None] = None):
        check_real_matrix(M)
        n, p = M.shape

        # Protect against absurd options
        if rank is not None:
            rank = max(1, min( rank, min(n, p) ))
        else:
            rank = min(n, p)

        # initialize initU, initV
        if initu is None or initv is None:
            initu, initv = self.get_initialized_U_V(M, rank)
        else:
            # both are present
            assert initu.shape[0] == n and initv.shape[0] == p and initu.shape[1] == rank and initv.shape[1] == rank, "Invalid initialization value"        
        
        if self.method == "v1":
            lambdas, initu, initv, iterlist = self.__decompose_v1(M, rank, initu, initv)
        else:
            lambdas, initu, initv, iterlist = self.__decompose_v2(M, rank, initu, initv)

        
        idx = lambdas.argsort()[::-1]   # sort the singular values in decreasing order
        lambdas = lambdas[idx]
        initu = initu[:, idx]
        initv = initv[:, idx]   # rearrage the singular values in the same order


        if self.verbose:
            self.sanity_check(M, lambdas, (iterlist >= self.maxiter))
        
        return SVDResult(initu, lambdas, initv, convergence = { 'iterations': iterlist })
