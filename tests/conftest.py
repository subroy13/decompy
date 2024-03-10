import pytest
import numpy as np
from decompy.interfaces import SVDResult, PCAResult, LSNResult, RankFactorizationResult


@pytest.fixture
def sample_matrix():
    n, p = 5, 4
    return [
        np.arange(n * p)
        .reshape((n, p))
        .astype("float64")  # TODO: need to able to remove typecasting
    ]


@pytest.fixture
def sample_svdresult():
    n, p, r = 10, 7, 3
    U = np.arange(n * r).reshape((n, r))
    V = np.arange(p * r).reshape((p, r))
    D = np.array([5, 3, 2])
    return SVDResult(U, D, V, convergence={"iterations": 100, "error": 1e-2})


@pytest.fixture
def sample_pcaresult():
    loc = np.array([1, 2, 3])
    evals = np.array([5, 4, 3])
    evecs = np.arange(15).reshape((5, 3))
    return PCAResult(loc, evals, evecs, convergence={"iterations": 100, "error": 1e-2})


@pytest.fixture
def sample_lsnresult():
    L = np.arange(15).reshape((5, 3))
    S = np.arange(15).reshape((5, 3))
    N = np.arange(15).reshape((5, 3))
    return LSNResult(L, S, N, convergence={"iterations": 100, "error": 1e-2})


@pytest.fixture
def sample_rankfactor():
    A = np.arange(15).reshape((5, 3))
    B = np.ones((4, 3))
    return RankFactorizationResult(A, B, convergence={"iterations": 100, "error": 1e-2})
