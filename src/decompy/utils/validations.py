import numpy as np


def check_real_matrix(X: np.ndarray):
    """The function checks if the input is a real matrix in the form of a numpy array.

    Parameters
    ----------
    X : np.ndarray
        a numpy array that represents a matrix.

    """
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert X.ndim == 2, "X must be a matrix"
    assert np.all(np.isreal(X)), "X must be a real matrix"


def check_binary_matrix(X: np.ndarray):
    """The function checks if a given numpy array is a binary matrix.

    Parameters
    ----------
    X : np.ndarray
        a numpy array representing a binary matrix. The function `check_binary_matrix` checks if the input
    matrix satisfies the following conditions:

    """
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert X.ndim == 2, "X must be a matrix"
    assert np.all(np.logical_or(X == 0, X == 1)), "X must be a binary matrix"


def is_increasing(a):
    """The function checks if a given array is in increasing order.

    Parameters
    ----------
    a
        a is a one-dimensional numpy array containing a sequence of numbers. The function is_increasing
    checks if the sequence is strictly increasing, meaning that each element in the sequence is greater
    than the previous element. The function returns True if the sequence is strictly increasing and
    False otherwise.

    Returns
    -------
        The function `is_increasing(a)` returns a boolean value indicating whether the input array `a` is
    increasing or not. It returns `True` if all the elements in the array are in increasing order, and
    `False` otherwise.

    """
    return np.all(a[:-1] <= a[1:])


def is_decreasing(a):
    """The function checks if all elements in a given array are in decreasing order.

    Parameters
    ----------
    a
        The parameter `a` is a one-dimensional numpy array of numbers. The function `is_decreasing` checks
    if the elements in the array are in decreasing order, meaning that each element is greater than or
    equal to the element that comes after it. The function returns `True` if the array is

    Returns
    -------
        The function `is_decreasing(a)` returns a boolean value indicating whether the input array `a` is
    in decreasing order or not. If `a` is in decreasing order, the function returns `True`, otherwise it
    returns `False`.

    """
    return np.all(a[:-1] >= a[1:])
