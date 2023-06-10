from typing import Literal
import numpy as np

class PCAResult:

    def __init__(self, loc: np.ndarray, eval: np.ndarray, evec: np.ndarray, **kwargs) -> None:
        assert len(loc.shape) == 1 and len(eval.shape) == 1 and len(evec.shape) == 2, "Mismatched shape"
        assert loc.shape[0] >= eval.shape[0] and eval.shape[0] == evec.shape[1], "Mismatched shape"
        self.loc = loc
        self.eval = eval
        self.evec = evec
        self.metrics = kwargs

    def location(self):
        return self.loc.reshape(-1)
    
    def eigen_values(self, as_matrix: bool = True):
        if as_matrix:
            return np.diag(self.eval.shape(-1))
        else:
            return self.eval.shape(-1)
        
    def eigen_vectors(self):
        return self.evec
    
    def convergence_metrics(self):
        return self.metrics.get("convergence")
    
    def cumulative_variance(self, type: Literal["identity", "proportion"] = "identity", threshold: float = 1e-5):
        vecs = self.eigen_values(as_matrix = False)
        vecs[vecs < threshold] = 0
        if type == "identity":
            return np.cumsum(vecs ** 2)
        elif type == "proportion":
            return np.cumsum(vecs ** 2) / np.sum(vecs ** 2)
        else:
            raise ValueError("Invalid type")
        
    def estimated_rank(self, threshold: float = 1e-5):
        vecs = self.eigen_values(as_matrix = False)
        return (vecs >= threshold).sum()


