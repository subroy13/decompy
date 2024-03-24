import numpy as np
import pytest
from decompy.interfaces import SVDResult, LSNResult, RankFactorizationResult, PCAResult
from decompy.matrix_factorization import (
    RobustSVDDensityPowerDivergence,
    AlternatingDirectionMethod,
    AugmentedLagrangianMethod,
    ActiveSubspaceRobustPCA,
    DualRobustPCA,
    ExactAugmentedLagrangianMethod,
    FastPrincipalComponentPursuit,
    GrassmannAverage,
    InexactAugmentedLagrangianMethod,
    L1Filtering,
    LinearizedADMAdaptivePenalty,
    MEstimation,
    MixtureOfGaussianRobustPCA,
    OutlierPursuit,
    PrincipalComponentPursuit,
    RegulaizedL1AugmentedLagrangianMethod,
    SingularValueThresholding,
    SymmetricAlternatingDirectionALM,
    VariationalBayes
)


@pytest.fixture
def sample_matrix():
    n, p = 5, 4
    return np.random.random(n * p).reshape((n, p)).astype("float64")


class TestMatrixFactorization:

    def check_svdresult_sanity(self, res, n: int, p: int):
        assert isinstance(res, SVDResult)
        assert res.D.ndim == 1 and res.D.shape[0] <= min(n, p)
        r = res.D.shape[0]
        assert res.U.ndim == 2 and res.U.shape == (n, r)
        assert res.V.ndim == 2 and res.V.shape == (p, r)

    def check_pcaresult_sanity(self, res, n: int, p: int):
        assert isinstance(res, PCAResult)
        assert res.loc.ndim == 1 and res.loc.shape[0] == n
        assert res.evals.ndim == 1 and res.evals.shape[0] <= min(n, p)
        r = res.evals.shape[0]
        assert res.evecs.ndim == 2 and res.evecs.shape == (n, r)

    def check_lsnresult_sanity(self, res, n: int, p: int):
        assert isinstance(res, LSNResult)
        assert res.L.ndim == 2 and res.L.shape == (n, p)
        assert res.S is None or (res.S.ndim == 2 and res.S.shape == (n, p))
        assert res.N is None or (res.N.ndim == 2 and res.N.shape == (n, p))

    def check_rankfactor_sanity(self, res, n: int, p: int):
        assert isinstance(res, RankFactorizationResult)
        assert res.A.ndim == 2 and res.A.shape[0] == n and res.A.shape[1] <= min(n, p)
        r = res.A.shape[1]
        assert res.B.ndim == 2 and res.B.shape == (p, r)

    def test_adm(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = AlternatingDirectionMethod()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_alm(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = AugmentedLagrangianMethod()
        res = mod.decompose(X, rank=min(n, p))
        self.check_lsnresult_sanity(res, n, p)

    def test_asrpca(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = ActiveSubspaceRobustPCA()
        res = mod.decompose(X, k=min(n, p))
        self.check_lsnresult_sanity(res, n, p)

    def test_dualrpca(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = DualRobustPCA()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_ealm(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = ExactAugmentedLagrangianMethod()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_fpcp(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = FastPrincipalComponentPursuit()
        res = mod.decompose(X, initrank=1)
        self.check_svdresult_sanity(res, n, p)

    def test_ga(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = GrassmannAverage()
        res = mod.decompose(X, min(n, p))
        self.check_rankfactor_sanity(res, n, p)

    def test_ialm(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = InexactAugmentedLagrangianMethod()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_l1f(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = L1Filtering()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_ladmap(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = LinearizedADMAdaptivePenalty()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_mest(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = MEstimation()
        res = mod.decompose(X)
        self.check_pcaresult_sanity(res, n, p)

    def test_mog(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = MixtureOfGaussianRobustPCA()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_op(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = OutlierPursuit()
        res = mod.decompose(X, rank=min(n, p))
        self.check_lsnresult_sanity(res, n, p)

    def test_pcp(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = PrincipalComponentPursuit()
        res = mod.decompose(X)
        self.check_svdresult_sanity(res, n, p)

    def test_reg1alm(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = RegulaizedL1AugmentedLagrangianMethod()
        res = mod.decompose(X, r=min(n, p))
        self.check_rankfactor_sanity(res, n, p)

    def test_rsvddpd(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape

        # run version 1
        mod = RobustSVDDensityPowerDivergence(method="v1")
        res = mod.decompose(X, n, p)
        self.check_svdresult_sanity(res, n, p)

        # run version 2
        mod = RobustSVDDensityPowerDivergence(method="v2")
        res = mod.decompose(X, n, p)
        self.check_svdresult_sanity(res, n, p)

    def test_sadal(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = SymmetricAlternatingDirectionALM()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_svt(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = SingularValueThresholding()
        res = mod.decompose(X)
        self.check_lsnresult_sanity(res, n, p)

    def test_vprpca(self, sample_matrix):
        X = sample_matrix.copy()
        n, p = X.shape
        mod = VariationalBayes()
        res = mod.decompose(X)
        self.check_rankfactor_sanity(res, n, p)
