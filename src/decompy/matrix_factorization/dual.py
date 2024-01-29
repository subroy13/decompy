import numpy as np

from ..utils.validations import check_real_matrix

class DualRobustPCA:

    """
    This solves the robust PCA problem using dual formalization

    The Primal Robust PCA relaxation
        min \tau ( |A|_* + \lambda |E|_1 ) + 1/2 |(A,E)|_F^2
        subj  A+E = D

    The Dual problem
        max trace(D' * Y)
        subj max( |Y|_2, 1 \ \lambda |Y|_inf) <= 1

    Notes
    -----
    [1] Robust PCA: Exact Recovery of Corrupted Low-Rank Matrices via Convex Optimization", J. Wright et al., preprint 2009.
    [2] "Fast Convex Optimization Algorithms for Exact Recovery of a Corrupted Low-Rank Matrix", Z. Lin et al.,  preprint 2009.
    """

    def __init__(self, **kwargs) -> None:
        pass

    