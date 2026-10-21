import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v419 import Expr, initial_language
from flf_ecf.v420 import check_invariance, discover_invariants, reflection, scaling, translation

class TestV420(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-2.0, 2.0, 101).reshape(-1, 1)
        x = initial_language(1)[0]
        self.x = x
        self.square = Expr("square", (x,))

    def test_square_is_reflection_invariant(self):
        report = check_invariance(self.square, self.X, [reflection])
        self.assertTrue(report.passed, report)
        self.assertLessEqual(report.max_normalized_error, 1e-8)

    def test_raw_variable_is_not_reflection_invariant(self):
        report = check_invariance(self.x, self.X, [reflection])
        self.assertFalse(report.passed)
        self.assertEqual(report.failed_transform_indices, (0,))

    def test_square_is_not_translation_invariant(self):
        report = check_invariance(self.square, self.X, [translation(1.0)])
        self.assertFalse(report.passed)

    def test_candidate_reports_are_deterministic_and_ordered(self):
        reports = discover_invariants([self.square, self.x], self.X, [reflection])
        self.assertEqual([r.passed for r in reports], [True, False])
        self.assertEqual([r.checks for r in reports], [1, 1])

    def test_invalid_transform_output_rejected(self):
        with self.assertRaises(ValueError):
            check_invariance(self.square, self.X, [lambda X: X[:, :0]])

    def test_scaling_validation(self):
        with self.assertRaises(ValueError):
            scaling(0.0)
        report = check_invariance(self.square, self.X, [scaling(-1.0)])
        self.assertTrue(report.passed)

if __name__ == "__main__":
    unittest.main()
