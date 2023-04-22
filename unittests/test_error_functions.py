"""
unittests for TestErrorFunctions class
"""
from unittest import TestCase

import numpy as np
import tensorflow as tf
from numpy import testing

from optimizations.error_functions import ErrorFunctions


class TestErrorFunctions(TestCase):

    def test_ErrorFunctions(self):
        true_y_3 = np.array([-1.223, 0.934, -0.554])
        predicted_y_3 = np.array([-1.323, 0.0734, 10.154, -0.665])

        with self.assertRaises(RuntimeError):
            ErrorFunctions(true_y_3, predicted_y_3)

        with self.assertRaises(TypeError):
            ErrorFunctions("test", predicted_y_3)

    def test_mse(self):
        self.true_y_1 = np.array([0.223, 0.334, 0.554, 0.665])
        self.predicted_y_1 = np.array([0.323, 0.734, 0.154, 0.665])

        self.true_y_2 = np.array([-1.223, 0.934, -0.554, 0.165])
        self.predicted_y_2 = np.array([-1.323, 0.0734, 10.154, -0.665])

        test1 = ErrorFunctions(self.true_y_1, self.predicted_y_1)
        test2 = ErrorFunctions(self.true_y_2, self.predicted_y_2)
        testing.assert_array_almost_equal(test1.mse(), np.array([0.08250]))

        testing.assert_array_almost_equal(test2.mse(), np.array([29.02520]))

    def test_mae(self):
        self.true_y_1 = np.array([0.223, 0.334, 0.554, 0.665])
        self.predicted_y_1 = np.array([0.323, 0.734, 0.154, 0.665])

        self.true_y_2 = np.array([-1.223, 0.934, -0.554, 0.165])
        self.predicted_y_2 = np.array([-1.323, 0.0734, 10.154, -0.665])

        test1 = ErrorFunctions(self.true_y_1, self.predicted_y_1)
        test2 = ErrorFunctions(self.true_y_2, self.predicted_y_2)
        testing.assert_array_almost_equal(test1.mae(), np.array([0.22500]))

        testing.assert_array_almost_equal(test2.mae(), np.array([3.12465]))

    def test_bce(self):
        self.true_y_1 = np.array([1, 1, 1, 1])
        self.predicted_y_1 = np.array([0.323, 0.734, 0.154, 0.665])

        self.true_y_2 = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        self.predicted_y_2 = np.array([0.323, 0.0734, 0.154, 0.665, 0.01, 0.432,
                                       0.678, 0.02])

        test_1 = ErrorFunctions(self.true_y_1, self.predicted_y_1).bce()
        bce = tf.keras.losses.BinaryCrossentropy()
        tf_bce_1 = bce(self.true_y_1, self.predicted_y_1).numpy()
        testing.assert_array_almost_equal(test_1, tf_bce_1, decimal=5)

        test_2 = ErrorFunctions(self.true_y_2, self.predicted_y_2).bce()
        tf_bce_2 = bce(self.true_y_2, self.predicted_y_2).numpy()
        testing.assert_array_almost_equal(test_2, tf_bce_2, decimal=5)

        self.true_2 = np.array([1.323, 0.734, 0.154, 0.665])
        self.predicted_2 = np.array([0.323, 0.0734, 0.154, 0.665])

        with self.assertRaises(ValueError):
            ErrorFunctions(self.true_2, self.predicted_2).bce()

        self.true_3 = np.array([1, 1, 1, 1])
        self.predicted_3 = np.array([0.323, 1.734, 0.154, 0.665, 0.01])

        with self.assertRaises(RuntimeError):
            ErrorFunctions(self.true_3, self.predicted_3).bce()

        self.true_4 = np.array([-1, 1, 1, 1])
        self.predicted_4 = np.array([0.323, 1.734, 0.154, 0.665])

        with self.assertRaises(ValueError):
            ErrorFunctions(self.true_4, self.predicted_4).bce()
