from typing import Literal
import numpy as np

class SVDResult:

    def __init__(self, U: np.ndarray, D: np.ndarray, V: np.ndarray, **kwargs) -> None:
        assert len(D.shape) == 1 and len(U.shape) == 2 and len(V.shape) == 2, "Mismatched shape in outputted singular values"
        assert D.shape[0] == U.shape[1] and D.shape[0] == V.shape[1], "Mismatched shape in outputted singular values and vectors"
        self.U = U
        self.D = D
        self.V = V
        self.metrics = kwargs

    def singular_values(self, as_matrix: bool = True):
        if as_matrix:
            return np.diag(self.D)
        else:
            return self.D.reshape(-1)
        
    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        if type == "left":
            return self.U
        elif type == "right":
            return self.V
        elif type == "both":
            return self.U, self.V
        else:
            raise ValueError("Invalid type")
        
    def convergence_metrics(self):
        return self.metrics.get("convergence")
    
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

            
    

    