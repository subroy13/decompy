from typing import Literal
import numpy as np

class LSNResult:
    """
        The LSNResult class is a structure for holding the matrix decomposition
        in form of a low rank matrix L, a sparse matrix S and a small perturbation noise
        matrix N.
    """

    def __init__(self, L: np.ndarray, S: np.ndarray = None, N: np.ndarray = None, **kwargs) -> None:
        '''This is a constructor function that initializes the low rank matrix, sparse matrix, noise
        perturbation matrix, and metrics in the LSNResult class
        
        Parameters
        ----------
        L : np.ndarray
            a numpy ndarray representing the low rank matrix
        S : np.ndarray
            a sparse matrix that represents the sparse component of a given matrix. This is an optional parameter. If provided, it should be a 2D numpy
        array with the same shape as the low rank matrix L.
        N : np.ndarray
            The noise perturbation matrix, which is an optional parameter. If provided, it should be a 2D numpy
        array with the same shape as the low rank matrix L.
        
        '''
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
        
        self.L = L  # assign the low rank matrix
        self.S = S  # assign the sparse matrix
        self.N = N  # assign the noise perturbation matrix
        self.metrics = kwargs

    def convergence_metrics(self):
        '''This function returns the convergence metrics from a dictionary called "metrics".
        
        Returns
        -------
            The method `convergence_metrics` is returning the value of the key "convergence" from the `metrics`
        dictionary attribute of the object.
        
        '''
        return self.metrics.get("convergence")
    
    def singular_values(self, as_matrix: bool = True):
        '''This function computes the singular values of a matrix and returns them either as a diagonal matrix
        or a 1D array.
        
        Parameters
        ----------
        as_matrix : bool, optional
            A boolean parameter that specifies whether the singular values should be returned as a matrix
        (i.e., a diagonal matrix with the singular values on the diagonal) or as a 1D array. If `as_matrix`
        is `True`, the function returns the singular values as a matrix. If `as_matrix` is `False`, the function 
        returns the singular values as a vector.
        
        Returns
        -------
            The function `singular_values` returns the singular values of the matrix `self.L`. If the parameter
        `as_matrix` is set to `True`, it returns a diagonal matrix with the singular values on the diagonal.
        If `as_matrix` is set to `False`, it returns a 1D array with the singular values.
        
        '''
        s = np.linalg.svd(self.L, full_matrices=False, compute_uv=False)
        s = s.reshape(-1)
        if as_matrix:
            return np.diag(s)
        else:
            return s

    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        '''The function returns the left, right, or both singular vectors of a given matrix using the SVD
        method in numpy.
        
        Parameters
        ----------
        type : Literal["left", "right", "both"], optional
            The `type` parameter is a string literal that specifies which singular vectors to return. It can
        take one of three values: "left", "right", or "both". If "left" is specified, only the left singular
        vectors (U) are returned. If "right" is specified, only the right singular vectors (V) are returned.
        If "both" is specified, both singular values (U and V) are returned as a tuple
        
        Returns
        -------
            The function `singular_vectors` returns the left singular vectors, right singular vectors, or both
        depending on the value of the `type` parameter. If `type` is "left", the function returns the left
        singular vectors (matrix U), if `type` is "right", the function returns the right singular vectors
        (matrix V), and if `type` is "both", the function returns both
        
        '''
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
        '''The function calculates the cumulative variance of singular values and returns either the cumulative
        sum or proportion based on the specified type.
        
        Parameters
        ----------
        type : Literal["identity", "proportion"], optional
            The `type` parameter is a string literal that specifies the type of cumulative variance to be
        calculated. It can take two possible values: "identity" or "proportion". If "identity" is selected,
        the function returns the cumulative sum of the singular values. If "proportion" is selected, the function
        returns the cumulative proportion of variance explained by the singular values.
        threshold : float
            The threshold parameter is a float value that is used to set any singular values that are smaller
        than the threshold to zero. This is done to remove any noise or small variations in the data that
        may not be significant. The default value for threshold is 1e-5, which means that any singular
        
        Returns
        -------
            a numpy array containing the cumulative sum of the singular values of a matrix, either as an
        identity or as a proportion of the total sum of singular values. The type of output is determined by
        the `type` parameter, which can be either "identity" or "proportion". If the `type` parameter is not
        one of these two options, a ValueError is raised. The function
        
        '''
        vecs = self.singular_values(as_matrix = False)
        vecs[vecs < threshold] = 0
        if type == "identity":
            return np.cumsum(vecs)
        elif type == "proportion":
            return np.cumsum(vecs) / np.sum(vecs)
        else:
            raise ValueError("Invalid type")

    def estimated_rank(self, threshold: float = 1e-5):
        '''The function calculates the estimated rank of a matrix based on its singular values.
        
        Parameters
        ----------
        threshold : float
            The threshold parameter is a float value that is used to determine the minimum value of singular
        values that will be considered in the calculation of the estimated rank. Any singular value below
        this threshold will be considered as zero and will not be included in the rank calculation. The
        default value of the threshold is 1e-5
        
        Returns
        -------
            The function `estimated_rank` returns the estimated rank of the matrix represented by the instance
        of the `SVD` class. The estimated rank is calculated as the number of singular values that are
        greater than or equal to the specified threshold value. The function returns an integer value
        representing the estimated rank.
        
        '''
        vecs = self.singular_values(as_matrix = False)
        return (vecs >= threshold).sum()
    