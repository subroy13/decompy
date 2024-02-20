from typing import Union
import numpy as np

from ..utils.validations import check_real_matrix
from ..interfaces import RankFactorizationResult

class GrassmannAverage:
    """
    GRASSMANN_AVERAGE(X) returns a basis vector for the average one-dimensional
    subspace spanned by the data X. This is a N-by-D matrix containing N observations
    in R^D.

    GRASSMANN_AVERAGE(X, K) returns a D-by-K matrix of orthogonal basis vectors
    spanning a K-dimensional average subspace.

    References
    ----------
    [1] "Grassmann Averages for Scalable Robust PCA".  S. Hauberg, A. Feragen and M.J. Black. In CVPR 2014. http://ps.is.tue.mpg.de/project/Robust_PCA

    """

    def __init__(self, **kwargs):
        self.eps = kwargs.get("eps", 1e-10)
        self.em_iter = kwargs.get("em_iter", 3)
        self.trim_percent = kwargs.get("trim_percent", 0)
        assert self.trim_percent > 0 and self.trim_percent < 0.5, "trim_percent must be in (0, 0.5)"

    def _reorth(self, Q: np.ndarray, r: int, normr: Union[float, None] = None, index = None, alpha = 0.5, method = 0):
        """
            Reorthogonalize a vector using iterated Gram-Schmidt
            
            References
            ----------
            [1] Aake Bjorck, "Numerical Methods for Least Squares Problems", SIAM, Philadelphia, 1996, pp. 68-69.
            [2] J.~W. Daniel, W.~B. Gragg, L. Kaufman and G.~W. Stewart ``Reorthogonalization and Stable Algorithms Updating the Gram-Schmidt QR Factorization'', Math. Comp.,  30 (1976), no. 136, pp. 772-795.
            [3] B. N. Parlett, ``The Symmetric Eigenvalue Problem'', Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109
            [4] Rasmus Munk Larsen, DAIMI, 1998.
        """

        # initialize the parameters
        n, k1 = Q.shape
        if normr is None:
            normr = np.linalg.norm(r)
        
        if index is None:
            k = k1
            index = np.arange(k)
            simple = True
        else:
            k = index.shape[0]
            if k == k1 and index == np.arange(k):
                simple = True
            else:
                simple = False
        
        if k == 0 or n == 0:
            return (r, normr, 0, 0)
        
        s = np.zeros(k)

        normr_old = 0
        nre = 0

        while (normr < alpha * normr_old) or (nre == 0):
            if method == 1:
                if simple:
                    t = Q.T @ r
                    r -= Q @ t
                else:
                    t = Q[:, index].T @ r
                    r -= Q[:, index] @ t
            else:
                for i in index.tolist():
                    t = Q[:, i].T @ r
                    r -= Q[:, i] @ t
            s += t
            normr_old = normr
            normr = np.linalg.norm(r)
            nre += 1

            if nre > 4:
                # r is in span(Q) to full accuracy => accept r = 0 as the new vector.
                r = np.zeros(n)
                normr = 0
                return (r, normr, s, nre)
        return (r, normr, s, nre)


    def decompose(self, M: np.ndarray, K: Union[int, None] = None):
        check_real_matrix(M)
        X = M.copy()    # create a copy of the matrix to avoid side effects
        n, d = X.shape
        if K is None:
            K = 1
        if K > d:
            K = d
        vectors = np.zeros((d, K))
        
        converged = np.zeros(K)   # convergence metrics
        niter = np.zeros(k)

        for k in range(K):
            # compute the k-th principal component
            mu = np.random.random(size = d).reshape(-1) - 0.5
            mu = mu / np.linalg.norm(mu)

            # initialize using a few EM iterations
            for _ in range(self.em_iter):
                dots = X @ mu # (N,)
                mu = X.T @ dots # (D,)
                mu /= np.linalg.norm(mu)

            # now the grassmann average
            for iter in range(n):
                prev_mu = mu

                # compute angles and flip
                dot_signs = np.sign(X @ mu)   # (N, )

                # compute weighted grassmann mean / trimmed mean
                if self.trim_percent > 0:
                    weighted_product = X * dot_signs[:, np.newaxis]  # Element-wise multiplication with broadcasting
                    mu = np.mean(np.percentile(weighted_product, self.trim_percent, axis=0), axis=0)  # Compute trimmed mean
                else:
                    mu = X.T @ dot_signs  # (D, 1)
                mu = mu.reshape(-1)  # (D, )
                mu /= np.linalg.norm(mu)

                # check for convergence
                if np.max(np.abs(mu - prev_mu)) < self.eps:
                    break
            converged[k] = (iter < n)  # if max iteration is not reached, means converged
            niter[k] = iter

            # store the estimated vector
            # and possibly subtract it from data, and perform reorthonomralisation
            if k == 0:
                vectors[:, k] = mu
                X -= X @ mu @ mu.T
            else:
                mu = self._reorth(vectors[:k], mu, 1)
                mu /= np.linalg.norm(mu)
                vectors[:, k] = mu

                if k < (K- 1):
                    X -= X @ mu @ mu.T  # otherwise, no need to take difference

        return RankFactorizationResult(
            A = X @ vectors,
            B = vectors.T,
            convergence = {
                'converged': converged,
                'niter': niter 
            }
        )    
        

                

        

