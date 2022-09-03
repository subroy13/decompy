import numpy as np
from decompy import rpca
from decompy.rpca import PCP
from decompy.dpd import rSVDdpd

X = np.random.random((5, 4))
print('svd', np.linalg.svd(X)[1])

out1 = rSVDdpd(X)
print('rsvddpd', out1['S'], out1['convergence']['iterations'])

out2 = PCP(X)
print('PCP', out2['L'], out2['S'])


