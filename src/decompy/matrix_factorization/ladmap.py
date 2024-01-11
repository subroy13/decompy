from typing import Union
import numpy as np


class LinearizedADMAdaptivePenalty:
    """
        Linearized Alternating Direction Method with Adaptive Penalty

        It aims to solve the LRR problem
        min |Z|_*+lambda*|E|_2,1  s.t., X = XZ+E, where lambda 

        Notes
        ------
        [1] Linearized Alternating Direction Method with Adaptive Penalty for Low-Rank Representation, Lin et al., 2011 - http://arxiv.org/abs/1109.0367
        [2] Original MATLAB code created by Risheng Liu on 05/02/2011, rsliu0705@gmail.com

    """
    def __init__(self, **kwargs):
        pass

    def decompose(self, M: np.ndarray):
        pass