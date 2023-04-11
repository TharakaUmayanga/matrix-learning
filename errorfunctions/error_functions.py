"""
    This module contains ErrorFunctions class.
    """
import numpy as np


class ErrorFunctions:
    """
    This class contains error functions
    for neural networks and machine learning algorithms.
    """

    def __init__(self, input_no):
        self.input_no = input_no

    def sigmoid(self):
        """
        return sigmoid value for given input
        y=1/(1+e^(-x))
        """
        return 1 / (1 + np.exp(-self.input_no))
