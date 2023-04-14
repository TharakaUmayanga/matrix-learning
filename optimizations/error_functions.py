"""
this module contains ErrorFunctions class
"""
import numpy as np


class ErrorFunctions:
    """
    This class contains error functions
    for neural networks and machine learning algorithms.
    """

    def __init__(self, actual, predicted):
        if type(actual) != np.ndarray or type(predicted) != np.ndarray:
            raise TypeError(f"expected numpy.ndarray received {type(actual)}")
        elif len(actual) != len(predicted):
            raise RuntimeError("actual and predicted values should have same length\n"
                               f"actual len {len(actual)} predicted len "
                               f"{len(predicted)}")
        self.actual = actual
        self.predicted = predicted

    def mse(self):
        """
        :return: returns mean squared error for the input value
        """

        return np.sum(np.power(self.actual - self.predicted, 2) / len(self.actual))

    def mae(self):
        """
        :return: returns mean absolute error for input values
        """
        return np.sum(np.absolute(self.predicted - self.actual))/len(self.actual)

