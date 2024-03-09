from typing import Literal
import numpy as np


class PCAResult:
    """
    The PCAResult class is a structure for holding the matrix decomposition in form of location and the principal components.

    Parameters
    ----------
    loc : np.ndarray
        a numpy array representing the location of a point in space
    eval : np.ndarray
        A 1-dimensional numpy array representing the eigenvalues of a matrix.
    evec : np.ndarray
        `evec` is a 2D numpy array representing the eigenvectors of a matrix. Each column of the array represents an eigenvector.

    Attributes
    ----------
    metrics : dict
        a python dictionary object to hold arbitrary values related to convergence metrics of the relevant algorithm

    """

    def __init__(
        self, loc: np.ndarray, eval: np.ndarray, evec: np.ndarray, **kwargs
    ) -> None:
        """This is a constructor function that initializes the PCAResult object with given numpy arrays
        and optional keyword arguments.
        """
        assert (
            len(loc.shape) == 1 and len(eval.shape) == 1 and len(evec.shape) == 2
        ), "Mismatched shape"
        assert (
            loc.shape[0] >= eval.shape[0] and eval.shape[0] == evec.shape[1]
        ), "Mismatched shape"
        self.loc = loc
        self.eigval = eval
        self.evec = evec
        self.metrics = kwargs

    def location(self):
        """The function returns a flattened version of the "loc" attribute of the PCAResult.

        Returns
        -------
            The `location` method is returning a flattened version of the `loc` attribute of the object. The
        `reshape(-1)` method is used to convert the `loc` attribute into a one-dimensional array.

        """
        return self.loc.reshape(-1)

    def eigen_values(self, as_matrix: bool = True):
        """The function returns either the diagonal matrix of eigenvalues or the number of eigenvalues of a
        given matrix.

        Parameters
        ----------
        as_matrix : bool, optional
            A boolean parameter that determines whether the eigenvalues should be returned as a matrix or a 1D
        array. If `as_matrix` is `True`, the eigenvalues will be returned as a diagonal matrix, where the
        diagonal elements are the eigenvalues. If `as_matrix` is `False`,

        Returns
        -------
            If `as_matrix` is `True`, the function returns a diagonal matrix of the eigenvalues of the object
        `self`. If `as_matrix` is `False`, the function returns the number of eigenvalues of the object
        `self`.

        """
        if as_matrix:
            return np.diag(self.eigval.reshape(-1))
        else:
            return self.eigval.reshape(-1)

    def eigen_vectors(self):
        """The function returns the eigenvectors of the PCAResult.

        Returns
        -------
            The function `eigen_vectors` is returning the attribute `evec` of the object.

        """
        return self.evec

    def convergence_metrics(self):
        """This function returns the convergence metrics from a dictionary called "metrics".

        Returns
        -------
            The method `convergence_metrics` is returning the value of the key "convergence" from the `metrics`
        dictionary attribute of the object.

        """
        return self.metrics.get("convergence")

    def cumulative_variance(
        self,
        type: Literal["identity", "proportion"] = "identity",
        threshold: float = 1e-5,
    ):
        """The function calculates the cumulative variance of eigenvalues and returns either the cumulative sum
        of squared eigenvalues or the proportion of the cumulative sum of squared eigenvalues.

        Parameters
        ----------
        type : Literal["identity", "proportion"], optional
            The `type` parameter is a string literal that specifies the type of cumulative variance to be
        calculated. It can take two possible values: "identity" or "proportion". If "identity" is selected,
        the function returns the cumulative sum of the squared eigenvalues. If "proportion" is selected, the function
        returns the cumulative proportion of variances explained by the eigenvalues.
        threshold : float
            The threshold parameter is a float value that is used to filter out small eigenvalues. Any
        eigenvalue smaller than the threshold value will be set to zero. This is done to remove noise and
        numerical instability in the calculation of the cumulative variance.

        Returns
        -------
            a numpy array containing the cumulative sum of the squared eigenvalues of the matrix, either as an
        "identity" or "proportion" type. If "identity" is chosen, the function returns the cumulative sum of
        the squared eigenvalues. If "proportion" is chosen, the function returns the cumulative sum of the
        squared eigenvalues divided by the sum of all squared eigenvalues.

        """
        vecs = self.eigen_values(as_matrix=False)
        vecs[vecs < threshold] = 0
        if type == "identity":
            return np.cumsum(vecs**2)
        elif type == "proportion":
            return np.cumsum(vecs**2) / np.sum(vecs**2)
        else:
            raise ValueError("Invalid type")

    def estimated_rank(self, threshold: float = 1e-5):
        """The function calculates the estimated rank of a matrix based on its eigenvalues.

        Parameters
        ----------
        threshold : float
            The threshold parameter is a float value that is used to determine the minimum value of eigenvalues
        that will be considered in the calculation of the estimated rank. Any eigenvalue below this
        threshold will be considered as zero and will not be included in the calculation of the rank. The
        default value of the threshold is 1e-5

        Returns
        -------
            The function `estimated_rank` returns the estimated rank of the matrix based on the eigenvalues. It
        counts the number of eigenvalues that are greater than or equal to the threshold value (default is
        1e-5) and returns that count as the estimated rank.

        """
        vecs = self.eigen_values(as_matrix=False)
        return (vecs >= threshold).sum()
