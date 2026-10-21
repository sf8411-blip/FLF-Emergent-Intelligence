import unittest

from flf_ecf.v426 import ExperimentConfig, run_exchange_experiment


class V426Tests(unittest.TestCase):
    def test_report_has_three_controlled_arms(self):
        report = run_exchange_experiment(
            seeds=(3, 7),
            config=ExperimentConfig(cells=6, train_pairs=32,
                                    validation_pairs=32, test_pairs=64),
        )
        self.assertEqual(
            [s.condition for s in report.summaries],
            ["isolated", "neighbor_exchange", "randomized_exchange"],
        )
        self.assertEqual(len(report.outcomes), 2 * 6 * 3)
        self.assertEqual(len(report.paired_accuracy_gain_vs_isolated), 2)

    def test_compute_budget_is_equal_across_arms(self):
        report = run_exchange_experiment(
            seeds=(1,),
            config=ExperimentConfig(cells=6, candidates_per_cell=3, share_count=1,
                                    train_pairs=32, validation_pairs=32, test_pairs=64),
        )
        for outcome in report.outcomes:
            self.assertGreaterEqual(outcome.candidates_evaluated, 1)
            self.assertLessEqual(outcome.candidates_evaluated, 3)

    def test_results_are_reproducible(self):
        config = ExperimentConfig(cells=6, train_pairs=32,
                                  validation_pairs=32, test_pairs=64)
        a = run_exchange_experiment(seeds=(11, 12), config=config)
        b = run_exchange_experiment(seeds=(11, 12), config=config)
        self.assertEqual(a, b)

    def test_test_metrics_and_threshold_errors_are_bounded(self):
        report = run_exchange_experiment(
            seeds=(0,),
            config=ExperimentConfig(cells=4, train_pairs=32,
                                    validation_pairs=32, test_pairs=64),
        )
        for outcome in report.outcomes:
            self.assertGreaterEqual(outcome.test_accuracy, 0.0)
            self.assertLessEqual(outcome.test_accuracy, 1.0)
            self.assertGreaterEqual(outcome.validation_accuracy, 0.0)
            self.assertLessEqual(outcome.validation_accuracy, 1.0)
            self.assertGreaterEqual(outcome.threshold_error, 0.0)

    def test_rejects_invalid_seed_sets(self):
        with self.assertRaises(ValueError):
            run_exchange_experiment(seeds=())
        with self.assertRaises(ValueError):
            run_exchange_experiment(seeds=(1, 1))
        with self.assertRaises(ValueError):
            run_exchange_experiment(seeds=(-1,))

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(cells=3)
        with self.assertRaises(ValueError):
            ExperimentConfig(candidates_per_cell=100)
        with self.assertRaises(ValueError):
            ExperimentConfig(target_threshold=1.0)


if __name__ == "__main__":
    unittest.main()
