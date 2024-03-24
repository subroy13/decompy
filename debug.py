import numpy as np
import sys, os

sys.path.append("./src")

from decompy.matrix_factorization import OutlierPursuit

n, p = 5, 4
X = np.arange(n * p).reshape((n, p)).astype("float64")    

mod = OutlierPursuit()
res = mod.decompose(X)

print(res)