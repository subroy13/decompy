import numpy as np
from time import time


from decompy.algorithms.robust_pca import InexactAugmentedLagrangianMethod

X = np.random.rand(250, 40)
mod = InexactAugmentedLagrangianMethod()
start = time()
result = mod.decompose(X)
end = time()

# U, V = result.singular_vectors(type="both")
# s = result.singular_values(as_matrix=False)

print(f"Time taken {end - start} seconds")
# print(U.shape, V.shape, s.shape)
# print(s)

# # actual singular values
# _, ss, _ = np.linalg.svd(X, full_matrices=False)
# print(ss[:min(min(X.shape), 5)])