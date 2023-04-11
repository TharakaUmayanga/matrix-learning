"""
unittests for ErrorFunctions class
"""
from unittest import TestCase

from errorfunctions.error_functions import ErrorFunctions


class TestErrorFunctions(TestCase):

    def test_ErrorFunctions(self):
        type_check_1 = ErrorFunctions(12)
        self.assertEqual(12.0, type_check_1.input_no)

        type_check_2 = ErrorFunctions(12.0)
        self.assertEqual(12.0, type_check_2.input_no)

        with self.assertRaises(TypeError):
            ErrorFunctions("test")

    def test_sigmoid(self):
        test1 = ErrorFunctions(10).sigmoid()
        test2 = ErrorFunctions(0.3458).sigmoid()
        test3 = ErrorFunctions(0.00002).sigmoid()
        test4 = ErrorFunctions(-0.458).sigmoid()
        test5 = ErrorFunctions(-0.003).sigmoid()

        self.assertEqual(0.99995460, test1)
        self.assertEqual(0.58559872, test2)
        self.assertEqual(0.50000500, test3)
        self.assertEqual(0.38746039, test4)
        self.assertEqual(0.49925000, test5)