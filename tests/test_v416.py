import pathlib, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v416 import run_seed

class V416Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_seed(0)

    def test_temporal_scale_is_selected_from_data(self):
        self.assertEqual(self.r["order"], 2)

    def test_operational_states_are_induced(self):
        self.assertGreaterEqual(self.r["states"], 2)
        self.assertLess(self.r["prediction_nll"], self.r["baseline_nll"])

    def test_semantic_controls(self):
        self.assertLess(self.r["prediction_nll"], self.r["shuffled_nll"])
        self.assertGreater(self.r["control_reward"], self.r["action_ablated_reward"])

    def test_renaming_invariance(self):
        self.assertTrue(self.r["renaming_invariant"])

if __name__ == "__main__":
    unittest.main()
