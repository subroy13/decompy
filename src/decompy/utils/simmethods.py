import numpy as np

def generate_LSN(n, p, r, noise = None, outlier = None):
    '''The `generate_LSN` function generates a matrix `X` by multiplying two low rank random matrices `A`
    and `B`, and adding noise and outliers to the result.
    
    Parameters
    ----------
    n
        The parameter `n` represents the number of rows in the generated matrix `X`. It determines the size
    of the matrix along the vertical axis.
    p
        The parameter `p` represents the number of columns in the matrix `X`.
    r
        The parameter `r` represents the rank of the matrix `L`. It determines the number of linearly
    independent columns in the matrix `L`.
    noise
        The "noise" parameter controls the amount of random noise added to the generated matrix. It is a
    scalar value that determines the standard deviation of the noise. If the noise parameter is set to
    None, no noise will be added to the matrix.
    outlier
        The "outlier" parameter controls the presence of outliers in the generated data. It is a value
    between 0 and 1 that represents the probability of an outlier being present in each element of the
    data matrix. If the value is 0, no outliers will be present.
    
    Returns
    -------
        The function `generate_LSN` returns a matrix `X` which is the sum of three components: `L`, `S`,
    and `N`.
    
    '''
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
    X = L + S + N
    return X
