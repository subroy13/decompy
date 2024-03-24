import numpy as np
from decompy.interfaces import SVDResult, PCAResult, LSNResult, RankFactorizationResult


class TestSVDResultInterface:

    def test_singular_values_as_matrix(self, sample_svdresult: SVDResult):
        singular_values_matrix = sample_svdresult.singular_values(as_matrix=True)
        assert np.array_equal(singular_values_matrix, np.diag([5, 3, 2]))

    def test_singular_values_as_array(self, sample_svdresult: SVDResult):
        singular_values_array = sample_svdresult.singular_values(as_matrix=False)
        assert np.array_equal(singular_values_array, np.array([5, 3, 2]))

    def test_singular_vectors_left(self, sample_svdresult: SVDResult):
        singular_vectors_left = sample_svdresult.singular_vectors(type="left")
        assert np.array_equal(singular_vectors_left, np.arange(30).reshape((10, 3)))

    def test_singular_vectors_right(self, sample_svdresult: SVDResult):
        singular_vectors_right = sample_svdresult.singular_vectors(type="right")
        assert np.array_equal(singular_vectors_right, np.arange(21).reshape((7, 3)))

    def test_singular_vectors_both(self, sample_svdresult: SVDResult):
        singular_vectors_both = sample_svdresult.singular_vectors(type="both")
        assert np.array_equal(singular_vectors_both[0], np.arange(30).reshape((10, 3)))
        assert np.array_equal(singular_vectors_both[1], np.arange(21).reshape((7, 3)))

    def test_convergence_metrics(self, sample_svdresult: SVDResult):
        convergence = sample_svdresult.convergence_metrics()
        assert convergence == {"iterations": 100, "error": 1e-2}

    def test_cumulative_variance_identity(self, sample_svdresult: SVDResult):
        cumulative_variance = sample_svdresult.cumulative_variance(type="identity")
        assert np.array_equal(cumulative_variance, np.array([5, 8, 10]))

    def test_cumulative_variance_proportion(self, sample_svdresult: SVDResult):
        cumulative_variance_proportion = sample_svdresult.cumulative_variance(
            type="proportion"
        )
        expected_variance_proportion = np.array([0.5, 0.8, 1])
        assert np.array_equal(
            cumulative_variance_proportion, expected_variance_proportion
        )

    def test_estimated_rank(self, sample_svdresult: SVDResult):
        estimated_rank = sample_svdresult.estimated_rank()
        assert estimated_rank == 3


class TestPCAResultInterface:

    def test_location(self, sample_pcaresult: PCAResult):
        location = sample_pcaresult.location()
        expected_location = np.array([1, 2, 3])
        assert np.array_equal(location, expected_location)

    def test_eigen_values_as_matrix(self, sample_pcaresult: PCAResult):
        eigen_values_matrix = sample_pcaresult.eigen_values(as_matrix=True)
        expected_matrix = np.diag([5, 4, 3])
        assert np.array_equal(eigen_values_matrix, expected_matrix)

    def test_eigen_values_as_array(self, sample_pcaresult: PCAResult):
        eigen_values_array = sample_pcaresult.eigen_values(as_matrix=False)
        expected_array = np.array([5, 4, 3])
        assert np.array_equal(eigen_values_array, expected_array)

    def test_eigen_vectors(self, sample_pcaresult: PCAResult):
        eigen_vectors = sample_pcaresult.eigen_vectors()
        expected_vectors = np.arange(15).reshape((5, 3))
        assert np.array_equal(eigen_vectors, expected_vectors)

    def test_convergence_metrics(self, sample_pcaresult: PCAResult):
        convergence = sample_pcaresult.convergence_metrics()
        assert convergence == {"iterations": 100, "error": 1e-2}

    def test_cumulative_variance_identity(self, sample_pcaresult: PCAResult):
        cumulative_variance = sample_pcaresult.cumulative_variance(type="identity")
        expected_variance = np.array([25, 41, 50])
        assert np.array_equal(cumulative_variance, expected_variance)

    def test_cumulative_variance_proportion(self, sample_pcaresult: PCAResult):
        cumulative_variance_proportion = sample_pcaresult.cumulative_variance(
            type="proportion"
        )
        expected_variance_proportion = np.array([0.5, 0.82, 1])
        assert np.allclose(cumulative_variance_proportion, expected_variance_proportion)

    def test_estimated_rank(self, sample_pcaresult: PCAResult):
        estimated_rank = sample_pcaresult.estimated_rank()
        assert estimated_rank == 3


class TestLSNResultInterface:

    def test_singular_values_as_matrix(self, sample_lsnresult: LSNResult):
        s = sample_lsnresult.singular_values(as_matrix=True)
        sexp = np.diag(np.linalg.svd(sample_lsnresult.L, compute_uv=False))
        assert np.allclose(s, sexp)

    def test_singular_values_as_array(self, sample_lsnresult: LSNResult):
        s = sample_lsnresult.singular_values(as_matrix=False)
        sexp = np.linalg.svd(sample_lsnresult.L, compute_uv=False)
        assert np.allclose(s, sexp)

    def test_singular_vectors_left(self, sample_lsnresult: LSNResult):
        U = sample_lsnresult.singular_vectors(type="left")
        Uexp, _, _ = np.linalg.svd(sample_lsnresult.L)
        assert np.allclose(U, Uexp)

    def test_singular_vectors_right(self, sample_lsnresult: LSNResult):
        V = sample_lsnresult.singular_vectors(type="right")
        _, _, Vexp = np.linalg.svd(sample_lsnresult.L)
        assert np.allclose(V, Vexp)

    def test_singular_vectors_both(self, sample_lsnresult: LSNResult):
        U, V = sample_lsnresult.singular_vectors(type="both")
        Uexp, _, Vexp = np.linalg.svd(sample_lsnresult.L)
        assert np.allclose(U, Uexp) and np.allclose(V, Vexp)

    def test_convergence_metrics(self, sample_lsnresult: LSNResult):
        convergence = sample_lsnresult.convergence_metrics()
        assert convergence == {"iterations": 100, "error": 1e-2}

    def test_cumulative_variance_identity(self):
        L = np.random.rand(10, 5)
        lsn = LSNResult(L=L)
        s = lsn.singular_values(as_matrix=False)
        cvar_id = lsn.cumulative_variance(type="identity")
        assert np.allclose(cvar_id, np.cumsum(s))

    def test_cumulative_variance_proportion(self):
        L = np.random.rand(10, 5)
        lsn = LSNResult(L=L)
        s = lsn.singular_values(as_matrix=False)
        cvar_prop = lsn.cumulative_variance(type="proportion")
        assert np.allclose(cvar_prop, np.cumsum(s) / np.sum(s))

    def test_estimated_rank(self):
        L = np.diag([1, 1, 1, 0.5, 0.1])
        lsn = LSNResult(L=L)
        rank = lsn.estimated_rank(threshold=0.51)
        assert rank == 3


class TestRankFactorizationResultInterface:

    def test_singular_values_as_matrix(
        self, sample_rankfactor: RankFactorizationResult
    ):
        s = sample_rankfactor.singular_values(as_matrix=True)
        L = sample_rankfactor.A @ sample_rankfactor.B.T
        sexp = np.diag(np.linalg.svd(L, compute_uv=False))
        assert np.allclose(s, sexp[:3, :3])

    def test_singular_values_as_array(self, sample_rankfactor: RankFactorizationResult):
        s = sample_rankfactor.singular_values(as_matrix=False)
        L = sample_rankfactor.A @ sample_rankfactor.B.T
        sexp = np.linalg.svd(L, compute_uv=False)
        assert np.allclose(s, sexp[:3])

    def test_singular_vectors_left(self, sample_rankfactor: RankFactorizationResult):
        U = sample_rankfactor.singular_vectors(type="left")
        Uexp, _, _ = np.linalg.svd(sample_rankfactor.A @ sample_rankfactor.B.T)
        assert np.allclose(U, Uexp[:, :3])

    def test_singular_vectors_right(self, sample_rankfactor: RankFactorizationResult):
        V = sample_rankfactor.singular_vectors(type="right")
        _, _, Vexp = np.linalg.svd(sample_rankfactor.A @ sample_rankfactor.B.T)
        assert np.allclose(V, Vexp[:3, :])

    def test_singular_vectors_both(self, sample_rankfactor: RankFactorizationResult):
        U, V = sample_rankfactor.singular_vectors(type="both")
        Uexp, _, Vexp = np.linalg.svd(sample_rankfactor.A @ sample_rankfactor.B.T)
        assert np.allclose(U, Uexp[:, :3]) and np.allclose(V, Vexp[:3, :])

    def test_convergence_metrics(self, sample_rankfactor: RankFactorizationResult):
        convergence = sample_rankfactor.convergence_metrics()
        assert convergence == {"iterations": 100, "error": 1e-2}

    def test_cumulative_variance_identity(
        self, sample_rankfactor: RankFactorizationResult
    ):
        s = sample_rankfactor.singular_values(as_matrix=False)
        cvar_id = sample_rankfactor.cumulative_variance(type="identity")
        assert np.allclose(cvar_id, np.cumsum(s))

    def test_cumulative_variance_proportion(
        self, sample_rankfactor: RankFactorizationResult
    ):
        s = sample_rankfactor.singular_values(as_matrix=False)
        cvar_prop = sample_rankfactor.cumulative_variance(type="proportion")
        assert np.allclose(cvar_prop, np.cumsum(s) / np.sum(s))

    def test_estimated_rank(self, sample_rankfactor: RankFactorizationResult):
        rank = sample_rankfactor.estimated_rank()
        assert rank == 3
