from typing import Literal
import numpy as np

class RankFactorizationResult:
    """
        The RankFactorizationResult class is a structure for holding the matrix decomposition
        in form of rank factorization.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, **kwargs):
        '''This function initializes a RankFactorizationResult with two numpy arrays A and B, and some optional metrics.
        
        Parameters
        ----------
        A : np.ndarray
            A is a numpy ndarray representing a matrix with 2 dimensions. It is one of the inputs to the
        constructor of a class.
        B : np.ndarray
            The parameter B is a numpy ndarray representing a matrix. The assert statements in the code ensure
        that B has two dimensions and that its second dimension matches the second dimension of the matrix
        A. The code then assigns the parameter B to the instance variable self.B.
        
        '''
        assert A.ndim == 2 and B.ndim == 2, "Mismatched shape"
        assert A.shape[1] == B.shape[1], "Mismatched shape"
        self.A = A
        self.B = B
        self.rank = A.shape[1]
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
        '''This function computes the singular values of a matrix product and returns them either as a diagonal
        matrix or a list.
        
        Parameters
        ----------
        as_matrix : bool, optional
            A boolean parameter that specifies whether the singular values should be returned as a matrix
        (i.e., a diagonal matrix with the singular values on the diagonal) or as a 1D array. If `as_matrix`
        is `True`, the function returns the singular values as a matrix. If `as
        
        Returns
        -------
            The function `singular_values` returns the singular values of the matrix product `A @ B.T` of the
        `MatrixFactorization` object. The singular values are computed using the `numpy.linalg.svd`
        function. The function returns either a diagonal matrix of singular values (if `as_matrix` is
        `True`) or a 1D array of singular values (if `as_matrix` is
        
        '''
        L = self.A @ self.B.T
        s = np.linalg.svd(L, full_matrices=False, compute_uv=False)
        s = s.reshape(-1)[:self.rank]
        if as_matrix:
            return np.diag(s)
        else:
            return s

    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        '''The function computes the singular vectors of a matrix product and returns either the left, right,
        or both singular vectors depending on the specified type.
        
        Parameters
        ----------
        type : Literal["left", "right", "both"], optional
            The `type` parameter is a string literal that specifies which singular vectors to return. It can
        take one of three values: "left", "right", or "both". If "left" is specified, only the left singular
        vectors are returned. If "right" is specified, only the right
        
        Returns
        -------
            The function `singular_vectors` returns either the left singular vector `U`, the right singular
        vector `V`, or both `U` and `V`, depending on the value of the `type` parameter. If `type` is not
        one of the valid options ("left", "right", "both"), a `ValueError` is raised.
        
        '''
        L = self.A @ self.B.T
        U, s, V = np.linalg.svd(L)
        U = U[:, self.rank]
        s = s[:, self.rank]
        V = V[:, self.rank]
        if type == "left":
            return U
        elif type == "right":
            return V
        elif type == "both":
            return U, V
        else:
            raise ValueError("Invalid type")

    def cumulative_variance(self, type: Literal["identity", "proportion"] = "identity"):
        '''The function calculates the cumulative variance of a matrix's singular values and returns either the
        cumulative sum or proportion of the singular values based on the specified type.
        
        Parameters
        ----------
        type : Literal["identity", "proportion"], optional
            The `type` parameter is a string literal that specifies the type of cumulative variance to be
        calculated. It can take two possible values: "identity" or "proportion". If "identity" is selected,
        the function returns the cumulative sum of the singular values. If "proportion" is selected the function
        returns the cumulative proportion of the variance explained by the singular values.
        
        Returns
        -------
            a numpy array containing the cumulative sum of the singular values of a matrix. The type of
        cumulative sum returned can be either "identity" or "proportion", depending on the argument passed
        to the function. If "identity" is passed, the function returns the cumulative sum of the singular
        values. If "proportion" is passed, the function returns the cumulative proportion of the singular
        values
        
        '''
        vecs = self.singular_values(as_matrix = False)
        if type == "identity":
            return np.cumsum(vecs)
        elif type == "proportion":
            return np.cumsum(vecs) / np.sum(vecs)
        else:
            raise ValueError("Invalid type")

    def estimated_rank(self):
        '''This function returns the rank of the result matrix decomposition.
        
        Returns
        -------
            the rank of the result matrix decomposition.
        
        '''
        return self.rank
