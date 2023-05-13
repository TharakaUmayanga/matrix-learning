import numpy
import numpy as np


class LinearRegression:
    """
    Linear regression is a linear approach to modeling the relationship between a scalar response and one or more
    """

    def __int__(self, x: numpy.ndarray, y: numpy.ndarray):
        self.x = x
        self.y = y

        if type(x) != np.ndarray or type(y) != np.ndarray:
            raise TypeError("x and y must be numpy.ndarray")
        elif len(x) != len(y):
            raise ValueError("x and y must have the same length")
        elif len(x) < 2:
            raise ValueError("x and y must have at least 2 elements")
        elif y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")

    def fit(self):
        """
        Fit the model using X as training data and y as target values
        :return: self
        """
        if self.x.ndim == 1:
            # simple linear regression y = ax + b

            pass
        pass

    x = np.array([1, 2, 3, 4, 5])
    x = x.reshape(-1, 1)
    print(x.ndim)
