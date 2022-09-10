import numpy as np
from scipy.sparse.linalg import svds as lansvd

from ..utils import checkIsMatrix

def OutlierPursuit(M, M_mask, lambdaval,options = {}):
    """
        Outlier Pursuit method
        - M is the input matrix to decompose
        - M_mask is binary matrix with mask(i,j) = 1 meaning M(i,j) = 1
        - lambdaval is a input tuning parameter
    """
    checkIsMatrix(M)
    checkIsMatrix(M_mask)
    X = np.copy(M)
    m,n = X.shape 
    assert M_mask.shape[0] == m, "Size of mask and M does not match"
    assert M_mask.shape[1]== n, "Size of mask and M does not match"

    # initialize settings from options
    rank_l = options.get("ini_rank", np.ceil(min(m, n) * 0.1))
    increaseK = options.get("increaseK", 10)
    f_svd = options.get("full_svd", True)
    maxiter = options.get("max_iter", 1000)

    delta = 1e-5
    mu_temp = 0.99 * np.linalg.norm(X)
    mu_bar = delta * mu_temp 
    eta = 0.9
    tol = 1e-6 * np.linalg.norm(X, 'fro')
    converged = False 

    # Temporary variables to hold intermediate results
    L_temp0 = np.zeros_like(X)
    L_temp1 = np.zeros_like(X)
    C_temp0 = np.zeros_like(X)
    C_temp1 = np.zeros_like(X)
    t_temp0 = 1
    t_temp1 = 1
    k = 0

    while not converged:
        YL = L_temp0 + (t_temp1 - 1)/t_temp0 * (L_temp0 - L_temp1)
        YC = C_temp0 + (t_temp1 - 1)/t_temp0 * (C_temp0 - C_temp1)

        M_diff = (YL + YC - M) * M_mask 
        
        GL = YL - 0.5 * M_diff 
        (L_new, rank_l) = iterate_L(GL, mu_temp/2, f_svd, increaseK)
        GC = YC - 0.5 * M_diff 
        C_new = iterate_C(GC, mu_temp * lambdaval / 2)
        
        t_new = (1 + np.sqrt(4*t_temp0**2 + 1)/2)
        mu_new = max(eta * mu_temp, mu_bar)

        # To decide if converged
        SL = 2*(YL - L_new) + (L_new + C_new - YL - YC)
        SC = 2 * (YC - C_new) + (L_new + C_new - YL - YC)
        if np.linalg.norm(SL, 'fro')**2 + np.linalg.norm(SC, 'fro')**2 <= tol**2:
            converged = True
        elif k > maxiter:
            converged = True
        else:
            L_temp1, L_temp0 = L_temp0, L_new
            C_temp1, C_temp0 = C_temp0, C_new
            t_temp1, t_temp0= t_temp0, t_new
            mu_temp = mu_new 
            k+=1

    return {
        'L': L_new,
        'C': C_new,
        'convergence': {
            'converged': (k >= maxiter),
            'iterations': k
        }
    }


def iterate_L(L, eps, f_svd, starting_K):
    if not f_svd:
        # TODO: Not supported
        raise NotImplementedError("Not supported")
    else:
        U, S, V = np.linalg.svd(L)
        rank_out = 0

    for i in range(S.shape[0]):
        if S[i] > eps:
            S[i] -= eps
        elif S[i] < -eps:
            S[i] += eps
        else:
            S[i] = 0
    output = U @ np.diag(S) @ V.T
    return output

def iterate_C(C, eps):
    m, n = C.shape
    for i in range(n):
        tmp = C[:, i]
        norm_tmp = np.linalg.norm(tmp)
        if norm_tmp >eps:
            tmp -= (tmp * eps / norm_tmp)
        else:
            tmp = np.zeros(m, 1)
        C[:, i] = tmp
    return C