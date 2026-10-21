import unittest
from dataclasses import FrozenInstanceError

from flf_ecf.v427 import (
    CellLedger, Observation, ResourceBudget, ResourceUsage,
    make_threshold_message,
)


def sample_message(message_id="m1", threshold=0.5, observations=None):
    if observations is None:
        observations = (
            Observation(0.1, True), Observation(0.2, True),
            Observation(0.8, False), Observation(0.9, False),
        )
    return make_threshold_message(
        message_id=message_id, origin_cell="sender-A", sequence=1,
        threshold=threshold, scope="synthetic-distance-world",
        observations=observations,
    )


def ledger(cell_id="receiver-B", **kwargs):
    return CellLedger(cell_id, ResourceBudget(**{
        "compute_units": 100, "memory_items": 10,
        "inbound_bytes": 10000, "energy_units": 100.0, **kwargs,
    }))


class TestV427(unittest.TestCase):
    def test_message_is_immutable_and_digest_deterministic(self):
        m = sample_message()
        self.assertEqual(m.content_digest, sample_message().content_digest)
        with self.assertRaises(FrozenInstanceError):
            m.origin_cell = "spoofed"

    def test_sender_observations_are_not_receiver_validation(self):
        m = sample_message(observations=tuple(Observation(0.1, True) for _ in range(12)))
        local = tuple(Observation(0.1, False) for _ in range(12))
        result = ledger().receive_and_validate(m, local, required_accuracy=0.9)
        self.assertTrue(result.accepted)
        self.assertFalse(result.validation.passed)
        self.assertEqual(result.validation.accuracy, 0.0)
        self.assertNotEqual(result.validation.validation_dataset_digest, m.content_digest)

    def test_passes_only_finite_local_test(self):
        local = tuple([Observation(0.1, True)] * 5 + [Observation(0.8, False)] * 5)
        result = ledger().receive_and_validate(sample_message(threshold=0.5), local,
                                               required_accuracy=0.9, min_samples=8)
        self.assertTrue(result.validation.passed)
        self.assertEqual(result.validation.accuracy, 1.0)
        self.assertEqual(result.validation.reason, "passed_finite_local_test")

    def test_insufficient_samples_fail_closed(self):
        result = ledger().receive_and_validate(sample_message(), [Observation(0.1, True)] * 3,
                                               min_samples=8)
        self.assertTrue(result.accepted)
        self.assertFalse(result.validation.passed)
        self.assertEqual(result.validation.reason, "insufficient_independent_local_samples")

    def test_duplicate_does_not_consume_more_resources(self):
        l = ledger()
        m = sample_message()
        first = l.receive_and_validate(m, [Observation(0.1, True)] * 8)
        before = l.usage
        second = l.receive_and_validate(m, [Observation(0.1, True)] * 8)
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "duplicate_message_id")
        self.assertEqual(l.usage, before)

    def test_resource_debit_is_atomic_when_compute_insufficient(self):
        l = CellLedger("small", ResourceBudget(5, 2, 10000, 10.0))
        before = l.usage
        self.assertFalse(l.consume(compute=6, memory=1, inbound_bytes=10, energy=1.0))
        self.assertEqual(l.usage, before)

    def test_message_size_budget_rejects_without_mutation(self):
        l = CellLedger("small", ResourceBudget(100, 2, 1, 100.0))
        before = l.usage
        result = l.receive_and_validate(sample_message(), [Observation(0.1, True)] * 8)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "inbound_byte_budget_exhausted")
        self.assertEqual(l.usage, before)
        self.assertEqual(l.messages, {})

    def test_memory_budget_is_enforced(self):
        l = CellLedger("small", ResourceBudget(100, 1, 10000, 100.0))
        m1 = sample_message("m1")
        m2 = sample_message("m2")
        local = [Observation(0.1, True)] * 8
        self.assertTrue(l.receive_and_validate(m1, local).accepted)
        before = l.usage
        result = l.receive_and_validate(m2, local)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "memory_budget_exhausted")
        self.assertEqual(l.usage, before)

    def test_negative_or_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            Observation(float("nan"), True)
        with self.assertRaises(ValueError):
            ResourceBudget(1, 1, 1, float("inf"))
        with self.assertRaises(ValueError):
            sample_message(threshold=-0.1)

    def test_invalid_message_schema_rejected(self):
        from flf_ecf.v427 import EvidenceMessage, ThresholdClaim
        with self.assertRaises(ValueError):
            EvidenceMessage("bad", "wrong-schema", "distance_threshold_v1",
                            ThresholdClaim(0.2, "scope"), (Observation(0.1, True),),
                            "cell", 0, 0, 0.0)

    def test_provenance_links_are_unique_and_not_self_referential(self):
        with self.assertRaises(ValueError):
            make_threshold_message(message_id="m1", origin_cell="A", sequence=0,
                                   threshold=0.1, scope="s",
                                   observations=[Observation(0.1, True)],
                                   parent_message_ids=("m2", "m2"))
        with self.assertRaises(ValueError):
            make_threshold_message(message_id="m1", origin_cell="A", sequence=0,
                                   threshold=0.1, scope="s",
                                   observations=[Observation(0.1, True)],
                                   parent_message_ids=("m1",))

    def test_usage_constructor_rejects_negative_counters(self):
        with self.assertRaises(ValueError):
            ResourceUsage(compute_units=-1)

    def test_validation_cost_is_charged_deterministically(self):
        l = ledger()
        local = [Observation(0.1, True)] * 9
        result = l.receive_and_validate(sample_message(), local,
                                       validation_compute_per_sample=2,
                                       validation_energy_per_sample=0.5)
        self.assertTrue(result.accepted)
        self.assertEqual(l.usage.compute_units, 18)
        self.assertEqual(l.usage.energy_units, 4.5)
        self.assertEqual(l.usage.memory_items, 1)
        self.assertEqual(l.usage.inbound_bytes, result.encoded_size_bytes)


if __name__ == "__main__":
    unittest.main()
