import numpy as np
import warnings
from ..utils import checkIsMatrix, is_decreasing

def rSVDdpd(M, alpha = 0.5, options = {}):
    assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"   # * Protect against absurd options
    checkIsMatrix(M)
    X = np.copy(M)
    n, p = X.shape

    # * Parse different options
    rank = options.get("rank", min(n, p))
    tol = options.get("tol", 1e-4)      # * Minimum tolerance level for the norm of X
    eps = options.get("eps", 1e-8)      # * Convergence criterion for change in left / right vectors
    maxiter = options.get("maxiter", 1e2)   # * Maximum number of iterations per singular value
    inneriter = options.get("inneriter", 5)   # * Number of inner fixed point iteration
    verbose = options.get("verbose", False)   # * Whether to peform sanity checks and output convergence issues

    # * Protect against absurb options
    rank = max(1, min(rank, min(n, p)))

    # initialize initU, initV
    initu = options.get("initu", np.random.random((n, rank)))
    initv = options.get("initv", np.random.random((p, rank)))
    for i in range(rank):
        initu[:, i] = initu[:, i] / np.linalg.norm(initu[:, i])
        initv[:, i] = initv[:, i] / np.linalg.norm(initv[:, i])
    lambdas = np.zeros(rank)
    iterlist = np.zeros(rank)

    # * normalize entries for better numerical precision
    scale_factor = np.quantile(X, [0.25, 0.75])
    X /= (scale_factor[1] - scale_factor[0])

    Xnorm = np.linalg.norm(X)

    for r in range(rank):
        curr_relnorm = np.linalg.norm(X) / Xnorm

        if (curr_relnorm > tol):
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
                    wmat = np.exp(-0.5 * alpha * (wmat ** 2) / sigma)
                    numer = np.multiply(wmat * X, curr_b).sum(axis = 1)
                    denom = np.multiply(wmat, curr_b**2).sum(axis = 1)
                    c = numer / denom 
                    left_iter += 1

                    # Check if fixed point criterion is met
                    fixed = (np.linalg.norm(c - curr_c) / np.linalg.norm(c) < eps) or (left_iter > inneriter)
                    
                # * Apply Gram-Schimdt Criterion
                if r > 0:
                    c -=((initu[:, :r] @ initu[:, :r].T) @ c)

                # Normalize
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
                    wmat = np.exp(-0.5 * alpha * (wmat ** 2) / sigma)
                    numer = np.multiply((wmat * X).T, curr_a).sum(axis = 1)
                    denom = np.multiply(wmat.T, curr_a**2).sum(axis = 1)
                    d = numer / denom 
                    right_iter += 1

                    # Check if fixed point criterion is met
                    fixed = (np.linalg.norm(d - curr_d) / np.linalg.norm(d) < eps) or (right_iter > inneriter)
                    
                # * Apply Gram-Schimdt Criterion
                if r > 0:
                    d -=((initv[:, :r] @ initv[:, :r].T) @ d)

                # Normalize
                curr_lambda = np.linalg.norm(d)
                curr_b = d / curr_lambda


                # STEP 3: Check if convergence is met
                niter += 1
                is_convl = (np.abs(curr_lambda - lambdas[r]) / (np.abs(lambdas[r]) + eps) < eps)
                is_conva = np.linalg.norm(curr_a - initu[:, r])
                is_convb = np.linalg.norm(curr_b - initv[:, r])

                converged = (niter > maxiter) or (is_convl and is_conva and is_convb)
                
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
    initu = initu[:, :r]    # Subset only the required column upto the rank
    initv = initv[:, :r]    

    if verbose:
        sanity_check(M, lambdas, (iterlist >= maxiter))
    return {
        'U': initu,
        'V': initv,
        'S': lambdas,
        'convergence': {
            'iterations': iterlist
        }
    }
    

def sanity_check(X, lambdas, maxit_reached):
    """
        Does some simple sanity check on the output
    """
    # * Check if maxit is reached
    if maxit_reached.sum() > 0:
        warnings.warn("Maximum iteration reached for singular values " + ",".join([str(i) for i in np.where(maxit_reached)[0] + 1]), stacklevel=1)
    if not is_decreasing(lambdas):
        warnings.warn("Singular values are not in sorted order")


        
