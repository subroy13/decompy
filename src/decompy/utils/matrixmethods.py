import numpy as np

def extract_submatrix(X: np.ndarray, rowmin: int, rowmax: int, colmin: int, colmax: int):
    '''The `extract_submatrix` function takes a matrix `X` and extracts a submatrix defined by the given
    row and column ranges, returning four submatrices A, B, C, and D.
    
    Parameters
    ----------
    X : np.ndarray
        The input matrix from which the submatrix will be extracted.
    rowmin : int
        The `rowmin` parameter represents the starting row index of the submatrix to be extracted.
    rowmax : int
        The parameter `rowmax` represents the maximum row index of the submatrix to be extracted.
    colmin : int
        The `colmin` parameter specifies the minimum column index to be included in the submatrix.
    colmax : int
        The parameter `colmax` represents the maximum column index of the submatrix to be extracted.
    
    Returns
    -------
        The function `extract_submatrix` returns four submatrices: A, B, C, and D.
    
    '''
    m, n = X.shape
    A = np.empty((rowmax - rowmin + 1, colmax - colmin + 1), dtype = X.dtype)
    B = np.empty((rowmax - rowmin + 1, n - colmax + colmin - 1), dtype = X.dtype)
    C = np.empty((m - rowmax + rowmin - 1, colmax - colmin + 1), dtype = X.dtype)
    D = np.empty((m - rowmax + rowmin - 1, n - colmax + colmin - 1), dtype = X.dtype)

    # add subsetting
    A = X[rowmin:(rowmax + 1), colmin:(colmax + 1)]
    B[:, :colmin] = X[rowmin:(rowmax + 1), :colmin]
    B[:, colmin:] = X[rowmin:(rowmax + 1), (colmax + 1):]
    C[:rowmin, :] = X[:rowmin, colmin:(colmax + 1)]
    C[rowmin:, :] = X[(rowmax + 1):, colmin:(colmax + 1)]
    D[:rowmin, :colmin] = X[:rowmin, :colmin]
    D[rowmin:, :colmin] = X[(rowmax + 1):, :colmin]
    D[:rowmin, colmin:] = X[:rowmin, (colmax + 1):]
    D[rowmin:, colmin:] = X[(rowmax + 1):, (colmax + 1):]

    return A, B, C, D


def g_inverse(X: np.ndarray, rank: int = 1):
    '''The `g_inverse` function performs a rank-SVD decomposition on an input matrix and returns its
    inverse.
    
    Parameters
    ----------
    X : np.ndarray
        The parameter `X` is a numpy array representing the input matrix for which we want to perform the
    SVD and inverse.
    rank : int, optional
        The `rank` parameter specifies the number of singular values and vectors to keep in the rank-SVD
    decomposition. It determines the level of approximation in the inverse calculation. By default, it
    is set to 1, meaning only the first singular value and vector will be used in the inverse
    calculation.
    
    Returns
    -------
        The function `g_inverse` returns the inverse of the rank-SVD decomposition of the input matrix `X`.
    
    '''
    U, s, Vt = np.linalg.svd(X, full_matrices = False)
    return Vt[:rank, :].T @ np.diag(1/s[:rank]) @ U[:, :rank].T


def accum_cv_metric(bcvs: np.ndarray, metric = "MSE"):
    '''The `accum_cv_metric` function calculates a specified metric (MSE, MAE, MAD, or L2) for a given
    array of cross-validation scores.
    
    Parameters
    ----------
    bcvs : np.ndarray
        The parameter "bcvs" is expected to be a numpy array containing the cross-validated metric scores.
    Each row of the array represents a different cross-validation fold, and each column represents a
    different metric score for a specific model or parameter setting.
    metric, optional
        The "metric" parameter is a string that specifies the type of metric to be used for calculating the
    scores. The available options are: MSE, MAE, MAD, and L2. The default value is MSE.
    
    Returns
    -------
        The function `accum_cv_metric` returns the calculated scores based on the input metric.
    
    '''
    if metric == "MSE":
        bcv_scores = np.mean(bcvs ** 2, axis = 0)
    elif metric == "MAE":
        bcv_scores = np.mean(np.abs(bcvs), axis = 0)
    elif metric == "MAD":
        bcv_scores = np.median(np.abs(bcvs), axis = 0)
    elif metric == "L2":
        bcv_var = np.median(np.absolute(bcvs - np.median(bcvs, axis = 0)), axis = 0) * 1.4826
        bcv_scores = np.mean(np.exp(-bcvs ** 2/ (2 * bcv_var**2)), axis = 0)
    else:
        raise ValueError("Invalid metric")
    return bcv_scores