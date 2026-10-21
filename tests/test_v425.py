import unittest
import numpy as np

from flf_ecf.v425 import Observation, discover_distance_relations


class V425Tests(unittest.TestCase):
    def test_reproducible_report(self):
        observations = [
            Observation(f"o{i}", (float(i) / 10, float(i % 2) / 10))
            for i in range(12)
        ]
        a = discover_distance_relations(observations, seed=7)
        b = discover_distance_relations(observations, seed=7)
        self.assertEqual(a, b)

    def test_report_exposes_bias_and_validation(self):
        observations = [
            Observation(f"o{i}", (float(i), float(i * 2)))
            for i in range(10)
        ]
        report = discover_distance_relations(observations, thresholds=(0.1, 0.5))
        self.assertEqual(report.observations, 10)
        self.assertEqual(report.feature_dimension, 2)
        for assessment in report.hypotheses:
            self.assertGreater(assessment.train_pairs, 0)
            self.assertGreater(assessment.validation_pairs, 0)
            self.assertTrue(any("human-specified" in c for c in assessment.caveats))

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            discover_distance_relations([Observation("a", (1.0,))] * 4)
        with self.assertRaises(ValueError):
            discover_distance_relations([
                Observation("a", (1.0,)), Observation("b", (2.0,)),
                Observation("c", (3.0,)), Observation("d", (float("nan"),))
            ])

    def test_mismatched_dimensions(self):
        items = [Observation("a", (1.0,)), Observation("b", (2.0,)),
                 Observation("c", (3.0,)), Observation("d", (4.0, 5.0))]
        with self.assertRaises(ValueError):
            discover_distance_relations(items)

    def test_parameter_validation(self):
        observations = [Observation(f"o{i}", (float(i),)) for i in range(8)]
        with self.assertRaises(ValueError):
            discover_distance_relations(observations, validation_fraction=1.0)
        with self.assertRaises(ValueError):
            discover_distance_relations(observations, thresholds=(0.1, 0.1))
        with self.assertRaises(ValueError):
            discover_distance_relations(observations, stability_tolerance=2.0)

    def test_threshold_support_is_required_for_stability(self):
        observations = [Observation(f"o{i}", (float(i * 100),)) for i in range(8)]
        report = discover_distance_relations(observations, thresholds=(0.0001,),
                                             min_pair_support=2)
        self.assertFalse(report.hypotheses[0].stable)

    def test_empty_thresholds_rejected(self):
        observations = [Observation(f"o{i}", (float(i),)) for i in range(8)]
        with self.assertRaises(ValueError):
            discover_distance_relations(observations, thresholds=())


if __name__ == "__main__":
    unittest.main()
