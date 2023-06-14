from typing import Literal
import numpy as np

class RankFactorizationResult:

    def __init__(self, A: np.ndarray, B: np.ndarray, **kwargs):
        assert A.ndim == 2 and B.ndim == 2, "Mismatched shape"
        assert A.shape[1] == B.shape[1], "Mismatched shape"
        self.A = A
        self.B = B
        self.rank = A.shape[1]
        self.metrics = kwargs

    def convergence_metrics(self):
        return self.metrics.get("convergence")

    def singular_values(self, as_matrix: bool = True):
        L = self.A @ self.B.T
        s = np.linalg.svd(L, full_matrices=False, compute_uv=False)
        s = s.reshape(-1)[:self.rank]
        if as_matrix:
            return np.diag(s)
        else:
            return s

    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        L = self.A @ self.B.T
        U, s, V = np.linalg.svd(L)
        U = U[:, self.rank]
        s = s[:, self.rank]
        V = V[:, self.rank]
        if type == "left":
            return U
        elif type == "right":
            return V
        elif type == "both":
            return U, V
        else:
            raise ValueError("Invalid type")

    def cumulative_variance(self, type: Literal["identity", "proportion"] = "identity"):
        vecs = self.singular_values(as_matrix = False)
        if type == "identity":
            return np.cumsum(vecs)
        elif type == "proportion":
            return np.cumsum(vecs) / np.sum(vecs)
        else:
            raise ValueError("Invalid type")

    def estimated_rank(self):
        return self.rank
