import numpy as np
import sys, os

sys.path.append("./src")

from decompy.matrix_factorization import L1Filtering

n, p = 5, 4
X = np.arange(n * p).reshape((n, p)).astype("float64")    

mod = L1Filtering()
res = mod.decompose(X)

print(res)