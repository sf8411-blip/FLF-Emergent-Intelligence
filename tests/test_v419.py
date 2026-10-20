import pathlib
import sys
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v419 import Expr, evaluate, initial_language, propose_operators, promote, validate_candidate

class TestV419(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(19419)
        self.X = self.rng.uniform(-2.0, 2.0, size=(600, 1))
        self.X_train, self.X_valid = self.X[:400], self.X[400:]
        self.y_train = self.X_train[:, 0] ** 2
        self.y_valid = self.X_valid[:, 0] ** 2

    def test_initial_language_has_only_raw_features(self):
        lang = initial_language(2)
        self.assertEqual(len(lang), 2)
        self.assertTrue(all(op.op == "var" for op in lang))

    def test_proposals_include_new_compositions(self):
        lang = initial_language(1)
        proposals = propose_operators(lang)
        self.assertTrue(any(op.op == "square" for op in proposals))
        self.assertNotIn(Expr("square", (lang[0],)), lang)

    def test_whitelist_evaluator(self):
        x = np.array([[-2.0], [3.0]])
        square = Expr("square", (Expr("var", index=0),))
        np.testing.assert_allclose(evaluate(square, x), [4.0, 9.0])
        with self.assertRaises(ValueError):
            evaluate(Expr("exec", ()), x)

    def test_true_operator_passes_heldout_gate(self):
        lang = initial_language(1)
        square = Expr("square", (lang[0],))
        report = validate_candidate(self.X_train, self.y_train, self.X_valid, self.y_valid,
                                    lang, square, min_relative_gain=0.02)
        self.assertTrue(report.accepted, report)
        self.assertGreaterEqual(report.relative_gain, 0.02)
        self.assertIn(square, promote(lang, report))

    def test_unhelpful_candidate_is_not_promoted(self):
        rng = np.random.default_rng(9)
        X = rng.uniform(-1, 1, size=(300, 1))
        y = X[:, 0]  # The raw-feature baseline fits this target exactly.
        lang = initial_language(1)
        candidate = Expr("sin", (lang[0],))
        report = validate_candidate(X[:200], y[:200], X[200:], y[200:], lang, candidate)
        self.assertFalse(report.accepted)
        self.assertEqual(report.reason, "baseline_already_exact")
        self.assertEqual(promote(lang, report), lang)

    def test_candidate_cannot_be_reintroduced(self):
        lang = initial_language(1)
        candidate = Expr("square", (lang[0],))
        with self.assertRaises(ValueError):
            validate_candidate(self.X_train, self.y_train, self.X_valid, self.y_valid,
                               lang + (candidate,), candidate)

    def test_nonfinite_features_are_rejected(self):
        X = self.X_train.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_candidate(X, self.y_train, self.X_valid, self.y_valid,
                               initial_language(1), Expr("square", (Expr("var", index=0),)))

if __name__ == "__main__":
    unittest.main()
