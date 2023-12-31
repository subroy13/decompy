import numpy as np

def extract_submatrix(X: np.ndarray, rowmin: int, rowmax: int, colmin: int, colmax: int):
    """
        Creates a partition matrix with rows from rowmin-rowcol and columns from colmin-colmax
        being extracted into 

        A   B
        C   D
    """
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
    """
        Perform rank-SVD and inverse
    """
    U, s, Vt = np.linalg.svd(X, full_matrices = False)
    return Vt[:rank, :].T @ np.diag(1/s[:rank]) @ U[:, :rank].T


def accum_cv_metric(bcvs: np.ndarray, metric = "MSE"):
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