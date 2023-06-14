import numpy as np

def check_real_matrix(X):
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert X.ndim == 2, "X must be a matrix"
    assert np.all(np.isreal(X)), "X must be a real matrix"

def check_binary_matrix(X):
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert X.ndim == 2, "X must be a matrix"
    assert np.all(X == 0 or X == 1), "X must be a binary matrix"


def is_increasing(a):
    return np.all(a[:-1] <= a[1:])


def is_decreasing(a):
    return np.all(a[:-1] >= a[1:])
