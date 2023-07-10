import numpy as np
from decompy.robust_svd import DensityPowerDivergence

# Load your data
data = np.arange(100).reshape(20,5).astype(np.float64)

# Perform data decomposition
algo = DensityPowerDivergence(alpha = 0.5)
result = algo.decompose(data)

# Access the decomposed components
U, V = result.singular_vectors(type = "both")
S = result.singular_values()
low_rank_component = U @ S @ V.T
sparse_component = data - low_rank_component


s2 = np.linalg.svd(data, compute_uv = False)
print(np.round(s2, 2))    # estimated by usual SVD
print(np.diag(S))    # estimated by robust SVD


data[1, 1] = 10000  # just change a single entry
s3 = np.linalg.svd(data, compute_uv = False)
print(np.round(s3, 2))   # usual SVD shoots up
s4 = algo.decompose(data).singular_values()
print(np.diag(np.round(s4, 2)))

