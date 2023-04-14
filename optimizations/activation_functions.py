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
        if type(input_no) != np.ndarray:
            raise TypeError(
                f"expected numpy.ndarray received {type(input_no)}")
        self.input_no = input_no

    def sigmoid(self):
        """
        return sigmoid value for given input
        y=1/(1+e^(-x))
        """
        return 1 / (1 + np.exp(-self.input_no))

    def relu(self):
        """
        f(x)=max(0.0, x).
        :return: return input if it's positive
        otherwise 0
        """
        return max(0.0, self.input_no)

    def leaky_relu(self):
        """
        f(x)=max(0.01*x , x).
        :return: return input * 0.01 if it's negative
        otherwise input no.
        """
        return self.input_no if self.input_no > 0 else self.input_no * 0.01

    def softmax(self):
        """
        :return: return softmax probabilities for
        given array
        """
        pass
