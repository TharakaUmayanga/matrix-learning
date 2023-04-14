"""
this module contains ErrorFunctions class
"""
import numpy as np


class ErrorFunctions:
    """
    This class contains error functions
    for neural networks and machine learning algorithms.
    """

    def __init__(self, input_no):
        if type(input_no) != np.ndarray:
            raise TypeError(f"expected numpy.ndarray received {type(input_no)}")
        self.input_no = input_no


