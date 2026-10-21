import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flf_ecf.v419 import Expr, variable
from flf_ecf.v420 import reflection, translation
from flf_ecf.v422 import search_counterexamples


class TestV422(unittest.TestCase):
    def test_finds_counterexample_for_non_invariant_expression(self):
        x = variable(0)
        report = search_counterexamples(x, reflection, -2.0, 2.0,
                                        n_samples=64, seed=11)
        self.assertTrue(report.refuted)
        self.assertGreater(report.max_normalized_error, 0.0)
        witness = report.counterexamples[0]
        self.assertAlmostEqual(witness.transformed_point[0],
                               -witness.input_point[0])

    def test_even_expression_survives_reflection_sample(self):
        x = variable(0)
        square = Expr("square", (x,))
        report = search_counterexamples(square, reflection, -2.0, 2.0,
                                        n_samples=100, tolerance=1e-12, seed=4)
        self.assertFalse(report.refuted)
        self.assertLessEqual(report.max_normalized_error, 1e-12)

    def test_deterministic_for_same_seed(self):
        x = variable(0)
        a = search_counterexamples(x, translation(1.0), -1.0, 1.0,
                                   n_samples=32, seed=42)
        b = search_counterexamples(x, translation(1.0), -1.0, 1.0,
                                   n_samples=32, seed=42)
        self.assertEqual(a, b)

    def test_per_feature_bounds(self):
        expr = Expr("add", (variable(0), variable(1)))
        report = search_counterexamples(expr, reflection, [-2.0, -1.0],
                                        [2.0, 1.0], n_samples=10, seed=1)
        self.assertEqual(report.searched_samples, 10)
        self.assertTrue(report.refuted)

    def test_rejects_invalid_bounds_and_limits(self):
        x = variable(0)
        with self.assertRaises(ValueError):
            search_counterexamples(x, reflection, 1.0, 1.0)
        with self.assertRaises(ValueError):
            search_counterexamples(x, reflection, -1.0, 1.0, n_samples=0)
        with self.assertRaises(ValueError):
            search_counterexamples(x, reflection, -1.0, 1.0, tolerance=float("nan"))

    def test_transform_shape_is_checked(self):
        x = variable(0)
        def bad_transform(X):
            return X[:, :0]
        with self.assertRaises(ValueError):
            search_counterexamples(x, bad_transform, -1.0, 1.0, n_samples=8)


if __name__ == "__main__":
    unittest.main()
