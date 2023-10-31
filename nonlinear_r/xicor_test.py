###############################################
# XICOR test class ############################
###############################################

import unittest
import numpy as np
from xicor import _check_inputs, _get_rank, _get_anti_rank, _get_numerator, _get_denominator, \
    xicor


class NonlinearCorrelationTester(unittest.TestCase):

    x = np.array([5., 4., 7.])
    y = np.array([5., 6., 4.])
    x_long = np.array([-0.985, 1.963, 0.461, 0.512, 0.183, -1.443, 1.415, -0.792, 0.468, -0.905,
                       -1.165, 0.008, -0.388, 0.33, -0.15, -0.916, 0.259, 0.079, 0.209, -0.565,
                       0.45, -0.091, 0.762, -1.404, 0.886, 1.579, 0.283, 0.439, -0.652, -0.998])
    y_long = np.array([0.946, 3.813, 0.289, 0.193, -0.219, 2.071, 2.003, 0.724, 0.154, 0.794,
                       1.242, 0.123, 0.489, -0.329, 0.49, 0.915, 0.149, 0.148, 0.124, 0.359,
                       0.494, 0.021, 0.662, 1.938, 0.778, 2.513, 0.029, 0.136, 0.061, 1.289])

    def test_all(self):
        self.test_input_error()
        self.test_input_ok()
        self.test_rank()
        self.test_anti_rank()
        self.test_numerator()
        self.test_denominator()
        self.test_nonlinear_r()
        self.test_nonlinear_p_value()

    def test_input_error(self):
        self.assertRaises(ValueError, _check_inputs, self.x, self.y[0:2])

    def test_input_ok(self):
        self.assertIsNone(_check_inputs(self.x, self.y))

    def test_rank(self):
        self.assertEqual(_get_rank(self.x).tolist(), [2, 1, 3])
        self.assertEqual(_get_rank(self.y).tolist(), [2, 3, 1])

    def test_anti_rank(self):
        self.assertEqual(_get_anti_rank(_get_rank(self.x)).tolist(), [2, 3, 1])
        self.assertEqual(_get_anti_rank(_get_rank(self.y)).tolist(), [2, 1, 3])

    def test_numerator(self):
        rank_y = _get_rank(self.y[np.argsort(self.x)])
        self.assertEqual(_get_numerator(rank_y), 6)

        rank_xx = _get_rank(self.x[np.argsort(self.x)])
        self.assertEqual(_get_numerator(rank_xx), 6)

        rank_yy = _get_rank(self.y[np.argsort(self.y)])
        self.assertEqual(_get_numerator(rank_yy), 6)

    def test_denominator(self):
        rank_y = _get_rank(self.y[np.argsort(self.x)])
        self.assertEqual(_get_denominator(rank_y), 8)

        rank_xx = _get_rank(self.x[np.argsort(self.x)])
        self.assertEqual(_get_denominator(rank_xx), 8)

        rank_yy = _get_rank(self.y[np.argsort(self.y)])
        self.assertEqual(_get_denominator(rank_yy), 8)

    def test_nonlinear_r(self):
        # For deterministic associations (i.e., no error), the max value of nonlinear_r is
        # (n - 2) / (n - 1);
        # In this test case, n=3 and thus r=0.25
        self.assertEqual(xicor(self.x, self.y)['statistic'], 0.25)
        self.assertEqual(xicor(self.x, self.x)['statistic'], 0.25)
        self.assertEqual(xicor(self.y, self.y)['statistic'], 0.25)
        # the following correlation value was taken from R package XICOR,
        # using the same input x_long and y_long
        self.assertAlmostEqual(xicor(self.x_long, self.y_long)['statistic'], 0.5995551)

    def test_nonlinear_p_value(self):
        self.assertAlmostEqual(xicor(self.x, self.y)['pvalue'], 0.2467814)
        self.assertAlmostEqual(xicor(self.x, self.x)['pvalue'], 0.2467814)
        self.assertAlmostEqual(xicor(self.y, self.y)['pvalue'], 0.2467814)
        self.assertAlmostEqual(xicor(self.x_long, self.y_long)['pvalue'], 1.038565e-07)