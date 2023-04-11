"""
    This module contains ActivationFunctions class.
    """
import numpy as np


class ActivationFunctions:
    """
    This class contains activation functions
    for neural networks and machine learning algorithms.
    """

    def __init__(self, input_no):
        if type(input_no) != float:
            try:
                input_no = float(input_no)
            except ValueError:
                raise TypeError(
                    f"Only integers and floats are available.\n"
                    f"expected int/float received {type(input_no)}")
        self.input_no = input_no

    def sigmoid(self):
        """
        return sigmoid value for given input
        y=1/(1+e^(-x))
        """
        return np.round(1 / (1 + np.exp(-self.input_no)), 8)
