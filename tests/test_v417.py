import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v417 import benchmark, run_seed

class TestV417(unittest.TestCase):
    def test_small_gate(self):
        summary, _ = benchmark(3)
        self.assertTrue(summary["gate_pass"], summary)

    def test_continuous_stream_discovers_boundaries_and_states(self):
        r = run_seed(0)
        self.assertGreaterEqual(r["boundary_f1"], 0.95)
        self.assertGreaterEqual(r["states"], 2)
        self.assertGreaterEqual(r["actuator_regions"], 2)

    def test_controls_break_semantics(self):
        r = run_seed(1)
        self.assertGreater(r["shuffled_actuator_rmse"], r["prediction_rmse"])
        self.assertGreater(r["random_segmentation_rmse"], r["prediction_rmse"])
        self.assertGreater(r["no_intervention_rmse"], r["prediction_rmse"])

    def test_affine_permutation_invariance(self):
        r = run_seed(2)
        rel = abs(r["invariant_rmse"] - r["prediction_rmse"])/r["prediction_rmse"]
        self.assertLessEqual(rel, 0.02)

if __name__ == "__main__":
    unittest.main()
