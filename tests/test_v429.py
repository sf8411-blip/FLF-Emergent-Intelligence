import unittest
import numpy as np
from flf_ecf.v429 import DiscoveryConfig, STRATEGIES, run_structure_discovery_benchmark

class V429Tests(unittest.TestCase):
    def test_all_strategies_run_on_shared_seeded_problem(self):
        cfg = DiscoveryConfig(seeds=3, samples=128, candidate_budget=12)
        report = run_structure_discovery_benchmark(cfg)
        self.assertEqual({r.strategy for r in report.summaries}, set(STRATEGIES))
        self.assertEqual(len(report.trials), 3 * len(STRATEGIES))
        for row in report.trials:
            self.assertTrue(np.isfinite(row.test_rmse))
            self.assertGreaterEqual(row.test_rmse, 0)
            self.assertGreaterEqual(row.target_components_found, 0)
            self.assertLessEqual(row.target_components_found, 2)

    def test_reproducible_for_fixed_seeds(self):
        cfg = DiscoveryConfig(seeds=2, samples=128, candidate_budget=10)
        self.assertEqual(run_structure_discovery_benchmark(cfg),
                         run_structure_discovery_benchmark(cfg))

    def test_rejects_invalid_config_and_strategy(self):
        with self.assertRaises(ValueError):
            DiscoveryConfig(seeds=0)
        with self.assertRaises(ValueError):
            run_structure_discovery_benchmark(
                DiscoveryConfig(seeds=1, samples=128), strategies=("quantum_magic",)
            )

    def test_candidate_budget_limits_unique_proposals(self):
        from flf_ecf.v429 import _candidate_order
        X = np.random.default_rng(3).normal(size=(40, 3))
        y = X[:, 0] ** 2 + np.sin(X[:, 1])
        for i, strategy in enumerate(STRATEGIES):
            candidates = _candidate_order(strategy, np.random.default_rng(i), 7, X, y)
            self.assertLessEqual(len(candidates), 7)
            self.assertEqual(len(set(candidates)), len(candidates))

if __name__ == "__main__":
    unittest.main()
