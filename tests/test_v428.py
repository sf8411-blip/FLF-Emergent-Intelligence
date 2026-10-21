import unittest

from flf_ecf.v428 import ExchangeConfig, run_typed_exchange_experiment


class V428Tests(unittest.TestCase):
    def small_config(self):
        return ExchangeConfig(cells=6, candidates_per_cell=3, share_count=1,
                              train_pairs=32, validation_pairs=32, test_pairs=64,
                              min_validation_samples=16, required_accuracy=0.85)

    def test_all_conditions_and_scenarios_are_reported(self):
        report = run_typed_exchange_experiment(
            seeds=(2, 5),
            config=self.small_config(),
            scenarios=("clean", "false_claims", "correlated_evidence", "distribution_shift"),
        )
        self.assertEqual(len(report.summaries), 12)
        self.assertEqual(len(report.trials), 2 * 4 * 6 * 3)
        for summary in report.summaries:
            self.assertGreaterEqual(summary.selection_coverage, 0.0)
            self.assertLessEqual(summary.selection_coverage, 1.0)
            if summary.selection_coverage == 0.0:
                self.assertIsNone(summary.mean_test_accuracy_when_selected)
            else:
                self.assertIsNotNone(summary.mean_test_accuracy_when_selected)
        self.assertEqual(
            {r.condition for r in report.summaries},
            {"isolated", "neighbor_exchange", "randomized_exchange"},
        )

    def test_resource_budget_is_not_exceeded(self):
        cfg = self.small_config()
        report = run_typed_exchange_experiment(seeds=(1,), config=cfg,
                                               scenarios=("clean",))
        for row in report.trials:
            self.assertLessEqual(row.candidates_considered, cfg.candidates_per_cell)
            self.assertLessEqual(row.resource_compute_used,
                                 cfg.candidates_per_cell * cfg.validation_pairs
                                 * cfg.validation_compute_per_sample)
            self.assertLessEqual(row.resource_bytes_used,
                                 cfg.candidates_per_cell * (512 + cfg.train_pairs * 64))

    def test_false_claims_are_rejected_by_receiver_local_validation(self):
        cfg = self.small_config()
        report = run_typed_exchange_experiment(seeds=tuple(range(8)), config=cfg,
                                               scenarios=("false_claims",))
        self.assertTrue(any(row.rejected_false_candidates > 0 for row in report.trials))

    def test_correlated_evidence_scenario_is_deterministic(self):
        cfg = self.small_config()
        a = run_typed_exchange_experiment(seeds=(7, 8), config=cfg,
                                          scenarios=("correlated_evidence",))
        b = run_typed_exchange_experiment(seeds=(7, 8), config=cfg,
                                          scenarios=("correlated_evidence",))
        self.assertEqual(a, b)

    def test_distribution_shift_is_explicitly_separated_from_validation(self):
        cfg = self.small_config()
        report = run_typed_exchange_experiment(seeds=(1, 2), config=cfg,
                                               scenarios=("distribution_shift",))
        self.assertTrue(all(row.selected_threshold is None or
                            0 < row.selected_threshold < 1 for row in report.trials))
        self.assertTrue(all(row.threshold_error >= 0 for row in report.trials))

    def test_rejects_invalid_seeds_and_scenarios(self):
        with self.assertRaises(ValueError):
            run_typed_exchange_experiment(seeds=(), config=self.small_config())
        with self.assertRaises(ValueError):
            run_typed_exchange_experiment(seeds=(1, 1), config=self.small_config())
        with self.assertRaises(ValueError):
            run_typed_exchange_experiment(seeds=(1,), config=self.small_config(),
                                          scenarios=("unknown",))


if __name__ == "__main__":
    unittest.main()
