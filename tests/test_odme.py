import unittest

import numpy as np

from src.odme import solve


class OdmeSolverTests(unittest.TestCase):
    def test_prior_consistent_with_observed_count_is_preserved(self):
        A = np.array([[1., 1.]])
        got = solve(A, np.array([10.]), np.array([4., 6.]), regularization=.05)
        np.testing.assert_allclose(got, [4., 6.], atol=1e-7)

    def test_nonnegative_solution_with_zero_count(self):
        got = solve(np.eye(2), np.array([0., 5.]), np.array([2., 5.]), regularization=.05)
        self.assertGreaterEqual(got.min(), 0.)
        self.assertLess(got[0], .1)


if __name__ == "__main__":
    unittest.main()
