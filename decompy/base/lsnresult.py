from typing import Literal
import numpy as np

class LSNResult:
    """
        Class for holding results in form of 
        Low Rank, Sparse and Noise Decomposition
    """

    def __init__(self, L: np.ndarray, S: np.ndarray = None, N: np.ndarray = None, **kwargs) -> None:
        assert L.ndim == 2, "Mismatched shape"
        if N is not None:
            assert N.ndim == 2, "Mismatched shape"
        if S is not None:
            assert S.ndim == 2, "Mismatched shape"
        
        m, n = L.shape
        if N is not None:
            assert N.shape[0] == m and N.shape[1] == n, "Mismatched shape"
        if S is not None:
            assert S.shape[0] == m and S.shape[1] == n, "Mismatched shape"
        
        self.L = L
        self.S = S
        self.N = N
        self.metrics = kwargs

    def convergence_metrics(self):
        return self.metrics.get("convergence")
    
    def singular_values(self, as_matrix: bool = True):
        s = np.linalg.svd(self.L, full_matrices=False, compute_uv=False)
        s = s.reshape(-1)
        if as_matrix:
            return np.diag(s)
        else:
            return s

    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        U, s, V = np.linalg.svd(self.L)
        if type == "left":
            return U
        elif type == "right":
            return V
        elif type == "both":
            return U, V
        else:
            raise ValueError("Invalid type")

    def cumulative_variance(self, type: Literal["identity", "proportion"] = "identity", threshold: float = 1e-5):
        vecs = self.singular_values(as_matrix = False)
        vecs[vecs < threshold] = 0
        if type == "identity":
            return np.cumsum(vecs)
        elif type == "proportion":
            return np.cumsum(vecs) / np.sum(vecs)
        else:
            raise ValueError("Invalid type")

    def estimated_rank(self, threshold: float = 1e-5):
        vecs = self.singular_values(as_matrix = False)
        return (vecs >= threshold).sum()
    