import unittest

from flf_ecf.v419 import Expr, variable
from flf_ecf.v420 import reflection
from flf_ecf.v423 import assess_hypothesis, run_revision_loop


class V423Tests(unittest.TestCase):
    def test_refuted_candidate_can_be_revised(self):
        x = variable(0)
        even = Expr("square", (x,))
        result = run_revision_loop(
            x, reflection, -2.0, 2.0,
            revise=lambda current, witnesses: even,
            max_revisions=2, n_samples=128, seed=17,
        )
        self.assertEqual(result.status, "survived_finite_search")
        self.assertEqual(result.final_expression, even)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].assessment.status, "refuted_discovery")

    def test_survival_is_not_called_proof(self):
        even = Expr("square", (variable(0),))
        report = assess_hypothesis(even, reflection, -3.0, 3.0,
                                   n_samples=100, seed=4)
        self.assertEqual(report.status, "survived_finite_search")
        self.assertFalse(report.counterexamples)
        self.assertNotIn("proved", report.status)

    def test_revision_can_stop(self):
        x = variable(0)
        result = run_revision_loop(
            x, reflection, -1.0, 1.0,
            revise=lambda current, witnesses: None,
            max_revisions=3, n_samples=32,
        )
        self.assertEqual(result.status, "revision_stopped")
        self.assertEqual(len(result.records), 1)

    def test_revision_budget_and_no_change(self):
        x = variable(0)
        with self.assertRaises(ValueError):
            run_revision_loop(x, reflection, -1.0, 1.0, lambda c, w: c,
                              max_revisions=-1)
        result = run_revision_loop(
            x, reflection, -1.0, 1.0,
            revise=lambda current, witnesses: current,
            max_revisions=2, n_samples=32,
        )
        self.assertEqual(result.status, "revision_no_change")

    def test_callback_type_checked(self):
        x = variable(0)
        with self.assertRaises(TypeError):
            run_revision_loop(x, reflection, -1.0, 1.0,
                              lambda current, witnesses: "not an expression",
                              max_revisions=1, n_samples=32)


if __name__ == "__main__":
    unittest.main()
