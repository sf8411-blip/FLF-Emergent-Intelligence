import unittest

from flf_ecf.v424 import CellGenome, DigitalCell, Experience, CellNetwork


def experience(eid="e1", *, validators=()):
    return Experience(
        experience_id=eid, claim="relation R holds in context C",
        context="C", outcome="observed_positive", evidence="finite observations",
        origin_cell="A", compute_cost=1, energy_cost=0.5,
        validated_by=tuple(validators),
    )


class V424Tests(unittest.TestCase):
    def test_genome_bounds(self):
        with self.assertRaises(ValueError):
            CellGenome(memory_limit=0)
        with self.assertRaises(ValueError):
            CellGenome(mutation_rate=1.1)

    def test_host_budget_is_hard_limit(self):
        cell = DigitalCell("A", CellGenome(compute_budget=2, energy_budget=1.0))
        self.assertTrue(cell.spend(1, 0.5))
        self.assertFalse(cell.spend(2, 0.1))
        self.assertEqual(cell.compute_remaining, 1)
        self.assertEqual(cell.energy_remaining, 0.5)

    def test_network_requires_neighbor_link(self):
        a, b = DigitalCell("A"), DigitalCell("B")
        a.remember(experience())
        network = CellNetwork([a, b])
        with self.assertRaises(ValueError):
            network.exchange("A", "B")
        network.connect("A", "B")
        report = network.exchange("A", "B")
        self.assertEqual((report.offered, report.accepted, report.rejected), (1, 1, 0))
        self.assertIn("e1", b.memory)

    def test_import_does_not_inherit_validation(self):
        a, b = DigitalCell("A"), DigitalCell("B")
        a.remember(experience(validators=("A", "C")))
        network = CellNetwork([a, b])
        network.connect("A", "B")
        network.exchange("A", "B")
        self.assertFalse(b.memory["e1"].verified)
        self.assertEqual(b.memory["e1"].validated_by, ())

    def test_memory_is_bounded_and_duplicates_are_rejected(self):
        cell = DigitalCell("A", CellGenome(memory_limit=1))
        self.assertTrue(cell.remember(experience("first")))
        self.assertFalse(cell.remember(experience("first")))
        self.assertTrue(cell.remember(experience("second")))
        self.assertEqual(tuple(cell.memory), ("second",))

    def test_unaffordable_experience_is_not_stored(self):
        cell = DigitalCell("B", CellGenome(compute_budget=1, energy_budget=0.1))
        self.assertFalse(cell.remember(experience()))
        self.assertEqual(cell.memory, {})


if __name__ == "__main__":
    unittest.main()
