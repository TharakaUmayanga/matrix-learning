"""
unittests for TestErrorFunctions class
"""
from unittest import TestCase

import numpy as np

from optimizations.error_functions import ErrorFunctions


class TestErrorFunctions(TestCase):

    def test_ErrorFunctions(self):
        self.true_y_1 = np.array([0.223, 0.334, 0.554, 0.665])
        self.predicted_y_1 = np.array([0.323, 0.734, 0.154, 0.665])

        self.true_y_2 = np.array([-1.223, 0.934, -0.554, 0.165])
        self.predicted_y_2 = np.array([-1.323, 0.0734, 10.154, -0.665])

        self.true_y_3 = np.array([-1.223, 0.934, -0.554])
        self.predicted_y_3 = np.array([-1.323, 0.0734, 10.154, -0.665])
        with self.assertRaises(RuntimeError):
            ErrorFunctions(self.true_y_3, self.predicted_y_3)

        with self.assertRaises(TypeError):
            ErrorFunctions("test",self.predicted_y_3)

