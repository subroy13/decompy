import numpy as np
from tqdm import tqdm


def rank_elbow(X: np.ndarray, svdfunc, maxrank=30):
    """
    Rank estimation using Elbow Method
    """
    U, s, V = svdfunc(
        X, rank=min(X.shape[0], X.shape[1], maxrank)
    )  # apply the svd function and get singular values
    ds = np.diff(s) / s[1:]
    minindex = np.argmin(ds)
    return minindex + 1


def rank_classical_ic(
    X: np.ndarray, svdfunc, ks=np.arange(20), criterion="AIC", verbose=False
):
    """
    Rank estimation using classical criterion:
    AIC - Akaike's Information Criteria (https://link.springer.com/chapter/10.1007/978-1-4612-1694-0_15)
    BIC - Bayesian Information Criteria
    """
    n, p = X.shape
    criterion = [criterion] if not isinstance(criterion, list) else criterion
    icvals = np.zeros((ks.shape[0], len(criterion)))

    # do the svd once
    maxrank = min(n, p, ks.max())
    Ufull, sfull, Vtfull = svdfunc(X, rank=maxrank)

    # iterator yielder
    ranger = tqdm(range(ks.shape[0])) if verbose else range(ks.shape[0])
    for i in ranger:
        rankk = ks[i]
        U, s, Vt = Ufull[:, :rankk], sfull[:rankk], Vtfull[:rankk, :]
        E = X - U @ np.diag(s) @ Vt
        Vsum = (E**2).mean()
        sigmahat = np.median(np.abs(E)) * 1.4826  # consistent estimate of sigma^2
        for j in range(len(criterion)):
            if criterion[j] == "AIC":
                icvals[i, j] = Vsum + rankk * sigmahat * 2 * (n + p - rankk) / (n * p)
            elif criterion[j] == "BIC":
                icvals[i, j] = Vsum + rankk * sigmahat * (n + p - rankk) * np.log(
                    n * p
                ) / (n * p)
            else:
                raise ValueError("Invalid criterion")
    ranks = (1 + np.argmin(icvals, axis=0)).reshape(-1)
    return ranks


def rank_bai_ng(
    X: np.ndarray, svdfunc, ks=np.arange(20), criterion="PC1", verbose=False
):
    """
    Rank estimation using information criterion by Bai and Ng.
    Reference - https://doi.org/10.1111/1468-0262.00273
    """
    n, p = X.shape
    criterion = [criterion] if not isinstance(criterion, list) else criterion
    icvals = np.zeros((ks.shape[0], len(criterion)))

    # do the svd once, and reuse it for partial SVD
    maxrank = min(n, p, ks.max())
    Ufull, sfull, Vtfull = svdfunc(X, rank=maxrank)

    # iterator yielder
    ranger = tqdm(range(ks.shape[0])) if verbose else range(ks.shape[0])
    for i in ranger:
        rankk = ks[i]
        U, s, Vt = Ufull[:, :rankk], sfull[:rankk], Vtfull[:rankk, :]
        E = X - U @ np.diag(s) @ Vt
        Vsum = (E**2).mean()
        sigmahat = np.median(np.abs(E)) * 1.4826  # consistent estimate of sigma^2
        for j in range(len(criterion)):
            if criterion[j] == "PC1":
                icvals[i, j] = Vsum + rankk * sigmahat * (n + p) / (n * p) * np.log(
                    (n * p) / (n + p)
                )
            elif criterion[j] == "PC2":
                icvals[i, j] = Vsum + rankk * sigmahat * (n + p) / (n * p) * np.log(
                    min(n, p)
                )
            elif criterion[j] == "PC3":
                icvals[i, j] = Vsum + rankk * sigmahat * np.log(min(n, p)) / min(n, p)
            elif criterion[j] == "IC1":
                icvals[i, j] = Vsum + rankk * (n + p) / (n * p) * np.log(
                    (n * p) / (n + p)
                )
            elif criterion[j] == "IC2":
                icvals[i, j] = Vsum + rankk * (n + p) / (n * p) * np.log(min(n, p))
            elif criterion[j] == "IC3":
                icvals[i, j] = Vsum + rankk * np.log(min(n, p)) / min(n, p)
            else:
                raise ValueError("Invalid criterion")

    ranks = (1 + np.argmin(icvals, axis=0)).reshape(-1)
    return ranks


def rank_DIC(X: np.ndarray, svdfunc, alpha=0.5, ks=np.arange(20), verbose=False):
    """
    Rank estimation using divergence information criterion
    Reference:
        - https://doi.org/10.1080/03610926.2017.1307405
    """
    n, p = X.shape
    DICvals = np.zeros(ks.shape[0])

    # do the svd once, and reuse it for partial SVD
    maxrank = min(n, p, ks.max())
    Ufull, sfull, Vtfull = svdfunc(X, rank=maxrank + 1)

    # iterator yielder
    ranger = tqdm(range(ks.shape[0])) if verbose else range(ks.shape[0])
    for i in ranger:
        rankk = ks[i]
        U, s, Vt = Ufull[:, :rankk], sfull[:rankk], Vtfull[:rankk, :]
        E = X - U @ np.diag(s) @ Vt
        sigma = np.median(np.abs(E)) * 1.4826  # consistent estimate of sigma^2

        # calculate the DPD
        sigmaconst = (2 * np.pi) ** (-alpha / 2) * sigma ** (-alpha)
        dpdt1 = 1 / np.sqrt(1 + alpha)
        dpdt2 = np.mean(np.exp(-alpha * E**2 / (2 * sigma**2)))
        penalty = (
            rankk * ((1 + alpha) / (1 + 2 * alpha)) ** (3 / 2) * (1 / n + 1 / p) / 2
        )
        DICvals[i] = sigmaconst * (dpdt1 - (1 + 1 / alpha) * dpdt2 + penalty)
    return ks[np.argmin(np.abs(np.diff(DICvals))) + 1]
