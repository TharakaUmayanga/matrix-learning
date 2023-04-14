"""
unittests for ActivationFunctions class
"""
from unittest import TestCase

import numpy as np
from numpy import testing

from optimizations.activation_functions import ActivationFunctions


class TestActivationFunctions(TestCase):

    def test_ActivationFunctions(self):
        input_1 = np.array([12])
        input_2 = np.array([12.0])
        type_check_1 = ActivationFunctions(input_1)
        self.assertEqual(input_1, type_check_1.input_no)

        type_check_2 = ActivationFunctions(input_2)
        self.assertEqual(input_2, type_check_2.input_no)

        with self.assertRaises(TypeError):
            ActivationFunctions("test")

    def test_sigmoid(self):
        test1 = ActivationFunctions(np.array([10])).sigmoid()
        test2 = ActivationFunctions(np.array([0.3458])).sigmoid()
        test3 = ActivationFunctions(np.array([0.00002])).sigmoid()
        test4 = ActivationFunctions(np.array([-0.458])).sigmoid()
        test5 = ActivationFunctions(np.array([-0.003])).sigmoid()
        test6 = ActivationFunctions(np.array([-0.003, 10, 0.3458])).sigmoid()

        testing.assert_array_almost_equal(np.array([0.99995460]), test1)
        testing.assert_array_almost_equal(np.array([0.58559872]), test2)
        testing.assert_array_almost_equal(np.array([0.50000500]), test3)
        testing.assert_array_almost_equal(np.array([0.38746039]), test4)
        testing.assert_array_almost_equal(np.array([0.49925000]), test5)
        testing.assert_array_almost_equal(np.array([0.49925000, 0.99995460, 0.58559872]), test6)

    def test_relu(self):
        test1 = ActivationFunctions(np.array([10])).relu()
        test2 = ActivationFunctions(np.array([0.3458])).relu()
        test3 = ActivationFunctions(np.array([-0.5468])).relu()
        test4 = ActivationFunctions(np.array([-5])).relu()

        testing.assert_equal(np.array([10]), test1)
        testing.assert_equal(np.array([0.3458]), test2)
        testing.assert_equal(np.array([0.0]), test3)
        testing.assert_equal(np.array([0.0]), test4)

    def test_leaky_relu(self):
        test1 = ActivationFunctions(np.array([10])).leaky_relu()
        test2 = ActivationFunctions(np.array([0.3458])).leaky_relu()
        test3 = ActivationFunctions(np.array([-0.5468])).leaky_relu()
        test4 = ActivationFunctions(np.array([-5])).leaky_relu()

        testing.assert_equal(np.array([10]), test1)
        testing.assert_equal(np.array([0.3458]), test2)
        testing.assert_equal(np.array([-0.005468]), test3)
        testing.assert_equal(np.array([-0.05]), test4)
