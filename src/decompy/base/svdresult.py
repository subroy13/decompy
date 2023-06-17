from typing import Literal
import numpy as np

class SVDResult:
    """
        The SVDResult class is a structure for holding the matrix decomposition
        in form of singular value decomposition.
    """

    def __init__(self, U: np.ndarray, D: np.ndarray, V: np.ndarray, **kwargs) -> None:
        '''This function initializes an output SVDReslt object with three input arrays and optional metrics.
        
        Parameters
        ----------
        U : np.ndarray
            a numpy array representing the left singular vectors of a matrix
        D : np.ndarray
            A 1-dimensional numpy array containing the singular values of a matrix.
        V : np.ndarray
            V is a 2D numpy array representing the right singular vectors of a matrix. It is part of the output
        of a singular value decomposition (SVD) and is used to reconstruct the original matrix.
        
        '''
        assert D.ndim == 1 and U.ndim == 2 and V.ndim == 2, "Mismatched shape in outputted singular values"
        assert D.shape[0] == U.shape[1] and D.shape[0] == V.shape[1], "Mismatched shape in outputted singular values and vectors"
        self.U = U
        self.D = D
        self.V = V
        self.metrics = kwargs

    def singular_values(self, as_matrix: bool = True):
        '''The function returns the singular values of a matrix either as a diagonal matrix or as a flattened
        array.
        
        Parameters
        ----------
        as_matrix : bool, optional
            A boolean parameter that specifies whether the singular values should be returned as a matrix (2D
        array) or as a 1D array. If `as_matrix` is `True`, the function returns a diagonal matrix with the
        singular values on the diagonal. If `as_matrix` is `False`,
        
        Returns
        -------
            The method `singular_values` returns either a diagonal matrix of singular values if `as_matrix` is
        `True`, or a 1D array of singular values if `as_matrix` is `False`.
        
        '''
        if as_matrix:
            return np.diag(self.D)
        else:
            return self.D.reshape(-1)
        
    def singular_vectors(self, type: Literal["left", "right", "both"] = "left"):
        '''The function `singular_vectors` returns the left, right, or both singular vectors of a matrix
        depending on the input type.
        
        Parameters
        ----------
        type : Literal["left", "right", "both"], optional
            The `type` parameter is a string literal that specifies which singular vectors to return. It can
        take one of three values: "left", "right", or "both". If "left" is specified, the method returns the
        left singular vectors (stored in the `U` attribute). If "
        
        Returns
        -------
            The `singular_vectors` method returns the left singular vectors if the `type` parameter is set to
        "left", the right singular vectors if `type` is set to "right", both left and right singular vectors
        if `type` is set to "both". If `type` is set to any other value, a `ValueError` is raised.
        
        '''
        if type == "left":
            return self.U
        elif type == "right":
            return self.V
        elif type == "both":
            return self.U, self.V
        else:
            raise ValueError("Invalid type")
        
    def convergence_metrics(self):
        '''This function returns the convergence metrics from a dictionary called "metrics".
        
        Returns
        -------
            The method `convergence_metrics` is returning the value of the key "convergence" from the `metrics`
        dictionary attribute of the object.
        
        '''
        return self.metrics.get("convergence")
    
    def cumulative_variance(self, type: Literal["identity", "proportion"] = "identity", threshold: float = 1e-5):
        '''The function calculates the cumulative variance of singular values and returns either the cumulative
        sum or proportion based on the specified type.
        
        Parameters
        ----------
        type : Literal["identity", "proportion"], optional
            The `type` parameter is a string literal that specifies the type of cumulative variance to be
        calculated. It can take two possible values: "identity" or "proportion". If "identity" is selected,
        the function returns the cumulative sum of the singular values. If "proportion" is selected
        threshold : float
            The threshold parameter is a float value that is used to set any singular values that are smaller
        than the threshold to zero. This is done to reduce the noise in the data and improve the accuracy of
        the results. Any singular values that are smaller than the threshold are considered to be
        insignificant and are therefore set
        
        Returns
        -------
            a numpy array containing the cumulative sum of the singular values of a matrix, either as an
        identity or as a proportion of the total sum of singular values. The type of output is determined by
        the `type` parameter, which can be either "identity" or "proportion". If the `type` parameter is not
        one of these two options, a ValueError is raised. 
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
        default value of the threshold is 1
        
        Returns
        -------
            The function `estimated_rank` returns the estimated rank of the matrix represented by the instance
        of the `SVD` class. The estimated rank is calculated as the number of singular values that are
        greater than or equal to the `threshold` value, which is set to `1e-5` by default. The function
        returns an integer value representing the estimated rank of the matrix.
        
        '''
        vecs = self.singular_values(as_matrix = False)
        return (vecs >= threshold).sum()

            
    

    