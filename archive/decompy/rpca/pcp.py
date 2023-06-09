import numpy as np
import warnings
from ..utils import checkIsMatrix, is_decreasing

def threshold_l1(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0)

def thresh_nuclear(M, thr):
    U, s, V = np.linalg.svd(M)
    dd = threshold_l1(s, thr)
    id = np.where(dd != 0)[0]
    s = dd[id]
    U = U[:, id]
    V = V[:, id]
    L = U @ np.diag(s) @ V.T
    return {
        's': s, 'U': U, 'V': V, 'L': L
    }



def PCP(M, lambdaval = None, muval = None, options = {}):
    """
        Principal Component Pursuit Method for Robust PCA

        Reference:
            - https://github.com/cran/rpca/blob/master/R/robustpca.R
    """
    checkIsMatrix(M)
    X = np.copy(M)
    n, p = X.shape
    if lambdaval is None:
        lambdaval = 1/np.sqrt(max(n, p))
    if muval is None:
        muval = n*p / (4 * np.abs(M).sum())
    
    delta = options.get("delta", 1e-7)
    maxiter = options.get("maxiter", 5e3)
    termnorm = delta * np.linalg.norm(X)
    verbose = options.get("verbose", False)

    S = np.zeros_like(X)
    Yimu = np.zeros_like(X)

    imu = 1/muval
    limu = lambdaval / muval

    niter = 0
    stats = []
    converged = False

    while (True):
        niter += 1
        Lsvd = thresh_nuclear(X - S + Yimu, imu)
        L = Lsvd['L']
        S = threshold_l1(X - L + Yimu, limu)
        MLS = X - L - S

        residnorm = np.linalg.norm(MLS)
        stats.append(residnorm)
        if verbose:
            print(f"Iteration: {niter}, Residual Norm: {residnorm}")
        
        converged = (residnorm < termnorm)
        if ((niter > maxiter) or converged):
            break

        Yimu += MLS

    finaldelta = residnorm * delta / termnorm
    if not converged:
        warnings.warn(f"RPCA using PCP approach did not converged after {niter} iterations.\nFinal delta: {finaldelta}\nPlease consider increasing maxiter.")

    return {
        'L': L,
        'S': S,
        'Lsvd': Lsvd,
        'convergence': {
            'converged': converged,
            'iterations': niter,
            'finaldelta': finaldelta,
            'alldelta': np.array(stats) * (delta / termnorm)
        }
    }

