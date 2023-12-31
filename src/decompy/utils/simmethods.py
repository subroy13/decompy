import numpy as np

def generate_LSN(n, p, r, noise = None, outlier = None):
    A = np.random.randn(n, r)
    B = np.random.randn(r, p)
    L = A @ B
    Lnorm = np.linalg.norm(L)   # we have ||L||^2 = E(||Z||^2) = np*var(Z)
    Lmax = np.abs(L).max()
    if noise:
        N = np.random.randn(n, p) * noise * Lnorm / np.sqrt(n*p)
    else:
        N = np.zeros((n,p))
    if outlier:
        S = np.random.binomial(1, outlier, size = n*p) * 5 * Lmax  # the outlier mask
        S = S.reshape(n,p)
    else:
        S = np.zeros((n, p))
    X = L + N + S
    return X
