import numpy as np
import itertools

from tqdm import tqdm
from tqdm.contrib import itertools as tqiter
from ..utils.matrixmethods import extract_submatrix, accum_cv_metric

def rank_gabriel(X: np.ndarray, svdfunc, ks = np.arange(1, 20), cvs = 100, verbose = False, metric = "MSE"):
    """
        Rank estimation using Gabriel style cross validation
        - http://www.numdam.org/item/JSFS_2002__143_3-4_5_0/
    """
    if any(ks > min(X.shape)):
        ks = np.arange(1, min(X.shape))

    metric = [metric] if not isinstance(metric, list) else metric
    bcvs = np.zeros((cvs, ks.shape[0]))

    # pick cvs many random indices
    row_indices = np.random.randint(0, X.shape[0], size = cvs)
    col_indices = np.random.randint(0, X.shape[1], size = cvs)

    # iterator yielder
    ranger = tqdm(range(cvs)) if verbose else range(cvs)

    for b in ranger:
        row_index = row_indices[b]
        col_index = col_indices[b]
        A, B, C, D = extract_submatrix(X, row_index, row_index, col_index, col_index)

        # do the SVD for the maximum rank
        Ufull, sfull, Vtfull = svdfunc(D, rank = ks.max())
        for i in range(ks.shape[0]):
            rankk = ks[i]
            U = Ufull[:, :rankk]
            s = sfull[:rankk]
            Vt = Vtfull[:rankk, :]

            Dplus = Vt.T @ np.diag(1/s) @ U.T

            Ahat = B @ Dplus @ C
            bcvs[b, i] = (A - Ahat)

    min_score_indices = []
    for met in metric:
        bcv_scores = accum_cv_metric(bcvs, met)
        min_score_indices.append(np.argmin(bcv_scores))
    ranks = ks[np.array(min_score_indices)].reshape(-1)
    return ranks

def rank_separate_rowcol(X: np.ndarray, svdfunc, ks = np.arange(1, 20), cvs = 100, verbose = False, metric = "MSE"):
    """
        Rank estimation using separate Row and Column deletion
        - https://www.jstor.org/stable/1267581
    """
    if any(ks > min(X.shape)):
        ks = np.arange(1, min(X.shape))
    bcvs = np.zeros((cvs, ks.shape[0]))
    metric = [metric] if not isinstance(metric, list) else metric

    # pick cvs many random indices
    row_indices = np.random.randint(0, X.shape[0], size = cvs)
    col_indices = np.random.randint(0, X.shape[1], size = cvs)

    # iterator yielder
    ranger = tqdm(range(cvs)) if verbose else range(cvs)

    for b in ranger:
        row_index = row_indices[b]
        col_index = col_indices[b]

        # row deletion
        _, _, _, R = extract_submatrix(X, row_index, row_index, 1, 0)
        Urow, srow, Vtrow = svdfunc(R, rank = ks.max())

        # col deletion
        _,_,_, C = extract_submatrix(X, 1, 0, col_index, col_index)
        Ucol, scol, Vtcol = svdfunc(C, rank = ks.max())

        for i in range(ks.shape[0]):
            rankk = ks[i]

            # estimate
            Xhat = Ucol[:, :rankk] @ np.diag(np.sqrt(srow[:rankk] * scol[:rankk])) @ Vtrow[:rankk, :]
            Xij = X[row_index, col_index]
            Xijhat = Xhat[row_index, col_index]

            bcvs[b, i] = (Xij - Xijhat)

    min_score_indices = []
    for met in metric:
        bcv_scores = accum_cv_metric(bcvs, met)
        min_score_indices.append(np.argmin(bcv_scores))
    ranks = ks[np.array(min_score_indices)].reshape(-1)
    return ranks

def rank_bcv(X: np.ndarray, svdfunc, ks = np.arange(1, 20), cvs = (None, None), verbose = False, metric = "MSE"):
    """"
        Rank estimation using Bi-cross validation technique
        Reference - https://doi.org/10.1214/08-AOAS227
    """
    if any(ks > min(X.shape)):
        ks = np.arange(1, min(X.shape))   # TODO: Check this
    cvx, cvy = cvs
    if cvx is None:
        cvx = int(X.shape[0] / 5)
    if cvy is None:
        cvy = int(X.shape[1] / 5)
    holdout_row = int(X.shape[0] / cvx)
    holdout_col = int(X.shape[1] / cvy)

    bcvs = np.zeros((cvx * cvy, ks.shape[0]))
    metric = [metric] if not isinstance(metric, list) else metric
    
    # iterator yielder
    if verbose:
        ranger = tqiter.product(range(cvx), range(cvy))
    else:
        ranger = itertools.product(range(cvx), range(cvy))

    for bi, bj in ranger:

        # pick cvs from holdouts
        row_start = holdout_row * bi
        col_start = holdout_col * bj
        row_end = holdout_row * (bi + 1) - 1
        col_end = holdout_col * (bj + 1) - 1
        A, B, C, D = extract_submatrix(X, row_start, row_end, col_start, col_end)
        U, s, Vt = svdfunc(D, rank = ks.max())

        for i in range(ks.shape[0]):
            rankk = ks[i]
            Dplus = Vt[:rankk, :].T @ np.diag(1/s[:rankk]) @ U[:, :rankk].T

            Ahat = B @ Dplus @ C
            index = (bi * cvy + bj)
            bcvs[index, i] = np.sum((A - Ahat)**2)

    min_score_indices = []
    for met in metric:
        bcv_scores = accum_cv_metric(bcvs, met)
        min_score_indices.append(np.argmin(bcv_scores))
    ranks = ks[np.array(min_score_indices)].reshape(-1)
    return ranks