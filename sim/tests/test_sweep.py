import unittest

from experiments.run_sweep import run_sweep


class SweepTests(unittest.TestCase):
    def test_default_sweep_uses_unique_seeds_and_initial_states(self):
        result = run_sweep(
            duration=0.02, replicates=2, omegas=[0.0, 0.2], rod_count=8,
            seed=12345, paired_seeds=False,
        )
        runs = result["runs"]
        self.assertEqual(len(runs), 4)
        self.assertEqual(len({row["seed"] for row in runs}), 4)
        self.assertEqual(len({row["initial_conditions_sha256"] for row in runs}), 4)
        self.assertEqual(result["seed_policy"], "unique per run")

    def test_paired_sweep_reuses_each_replicate_state_across_omegas(self):
        result = run_sweep(
            duration=0.02, replicates=2, omegas=[0.0, 0.2], rod_count=8,
            seed=12345, paired_seeds=True,
        )
        runs = result["runs"]
        self.assertEqual(len(runs), 4)
        by_pair = {}
        for row in runs:
            by_pair.setdefault(row["replicate"], []).append(row)
        self.assertEqual(len(by_pair[0]), 2)
        self.assertEqual(len(by_pair[1]), 2)
        for pair in by_pair.values():
            self.assertEqual(len({row["seed"] for row in pair}), 1)
            self.assertEqual(len({row["initial_conditions_sha256"] for row in pair}), 1)
        self.assertNotEqual(
            by_pair[0][0]["initial_conditions_sha256"],
            by_pair[1][0]["initial_conditions_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
