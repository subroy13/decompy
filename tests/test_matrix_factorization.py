import numpy as np
import pytest
from decompy.interfaces import SVDResult
from decompy.matrix_factorization import (
    RobustSVDDensityPowerDivergence
)

@pytest.fixture
def sample_matrix():
    n, p = 5, 4
    return np.arange(n * p).reshape((n, p)).astype('float64')


class TestMatrixFactorization:

    def check_svdresult_sanity(self, res, n: int, p: int):
        assert isinstance(res, SVDResult)
        assert res.D.ndim == 1 and res.D.shape[0] == min(n, p)
        assert res.U.ndim == 2 and res.U.shape == (n, min(n, p))
        assert res.V.ndim == 2 and res.V.shape == (p, min(n, p))

    def test_rsvddpd(self, sample_matrix):
        X = sample_matrix
        n, p = X.shape
        
        # run version 1
        mod = RobustSVDDensityPowerDivergence(method = "v1")
        res = mod.decompose(X, n, p)
        self.check_svdresult_sanity(res, n, p)

        # run version 2
        mod = RobustSVDDensityPowerDivergence(method = "v2")
        res = mod.decompose(X, n, p)
        self.check_svdresult_sanity(res, n, p)

        

