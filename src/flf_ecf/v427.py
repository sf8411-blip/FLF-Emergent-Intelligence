"""ECF v4.27: typed evidence, provenance, and receiver-side revalidation.

This module is deliberately narrow: it transports a distance-threshold relation
hypothesis with auditable observations. It does not treat sender confidence,
reputation, or validation labels as truth. All received claims start unverified.
No neural network or unrestricted executable payloads are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable, Mapping


SCHEMA_VERSION = "flf-evidence/1"
CLAIM_KIND = "distance_threshold_v1"


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _finite_nonnegative(value: float, field_name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Hard deterministic accounting limits; energy is a software proxy unless metered."""
    compute_units: int
    memory_items: int
    inbound_bytes: int
    energy_units: float

    def __post_init__(self) -> None:
        if min(self.compute_units, self.memory_items, self.inbound_bytes) < 0:
            raise ValueError("integer budget limits must be non-negative")
        _finite_nonnegative(self.energy_units, "energy_units")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    compute_units: int = 0
    memory_items: int = 0
    inbound_bytes: int = 0
    energy_units: float = 0.0

    def __post_init__(self) -> None:
        if min(self.compute_units, self.memory_items, self.inbound_bytes) < 0:
            raise ValueError("resource usage cannot be negative")
        _finite_nonnegative(self.energy_units, "energy_units")


@dataclass(frozen=True, slots=True)
class Observation:
    """One scalar-distance observation and an observed binary relation outcome."""
    distance: float
    relation_holds: bool

    def __post_init__(self) -> None:
        if not isfinite(self.distance) or self.distance < 0:
            raise ValueError("distance must be finite and non-negative")
        if not isinstance(self.relation_holds, bool):
            raise ValueError("relation_holds must be bool")


@dataclass(frozen=True, slots=True)
class ThresholdClaim:
    """A typed, bounded hypothesis: relation holds iff distance <= threshold."""
    threshold: float
    scope: str

    def __post_init__(self) -> None:
        if not isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        _require_nonempty(self.scope, "scope")


@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    """Immutable claim package. Sender-side status is intentionally not represented."""
    message_id: str
    schema_version: str
    claim_kind: str
    claim: ThresholdClaim
    observations: tuple[Observation, ...]
    origin_cell: str
    created_sequence: int
    declared_compute_cost: int
    declared_energy_cost: float
    parent_message_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.message_id, "message_id")
        _require_nonempty(self.origin_cell, "origin_cell")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if self.claim_kind != CLAIM_KIND:
            raise ValueError("unsupported claim_kind")
        if self.created_sequence < 0 or self.declared_compute_cost < 0:
            raise ValueError("sequence and compute cost must be non-negative")
        _finite_nonnegative(self.declared_energy_cost, "declared_energy_cost")
        if not self.observations:
            raise ValueError("at least one evidence observation is required")
        if len(set(self.parent_message_ids)) != len(self.parent_message_ids):
            raise ValueError("parent message IDs must be unique")
        if self.message_id in self.parent_message_ids:
            raise ValueError("message cannot be its own parent")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "claim_kind": self.claim_kind,
            "claim": {"threshold": self.claim.threshold, "scope": self.claim.scope},
            "observations": [
                {"distance": o.distance, "relation_holds": o.relation_holds}
                for o in self.observations
            ],
            "origin_cell": self.origin_cell,
            "created_sequence": self.created_sequence,
            "declared_compute_cost": self.declared_compute_cost,
            "declared_energy_cost": self.declared_energy_cost,
            "parent_message_ids": list(self.parent_message_ids),
        }

    @property
    def content_digest(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        return sha256(raw).hexdigest()

    @property
    def encoded_size_bytes(self) -> int:
        return len(json.dumps(
            {"message_id": self.message_id, **self.canonical_dict()},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Receiver-generated result; no imported result is accepted as authority."""
    message_id: str
    receiver_cell: str
    sample_count: int
    accuracy: float
    passed: bool
    required_accuracy: float
    validation_dataset_digest: str
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.message_id, "message_id")
        _require_nonempty(self.receiver_cell, "receiver_cell")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not isfinite(self.accuracy) or not 0 <= self.accuracy <= 1:
            raise ValueError("accuracy must be in [0, 1]")
        if not isfinite(self.required_accuracy) or not 0 <= self.required_accuracy <= 1:
            raise ValueError("required_accuracy must be in [0, 1]")
        _require_nonempty(self.validation_dataset_digest, "validation_dataset_digest")


@dataclass(frozen=True, slots=True)
class ReceiveResult:
    accepted: bool
    reason: str
    message_id: str
    encoded_size_bytes: int
    validation: ValidationResult | None = None


@dataclass
class CellLedger:
    """Single-threaded deterministic ledger. Calls are atomic at method boundaries."""
    cell_id: str
    budget: ResourceBudget
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    messages: dict[str, EvidenceMessage] = field(default_factory=dict)
    validations: dict[str, ValidationResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.cell_id, "cell_id")
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        u, b = self.usage, self.budget
        if (u.compute_units > b.compute_units or u.memory_items > b.memory_items
                or u.inbound_bytes > b.inbound_bytes or u.energy_units > b.energy_units):
            raise ValueError("initial resource usage exceeds budget")
        if len(self.messages) > b.memory_items:
            raise ValueError("message store exceeds memory budget")

    def can_afford(self, *, compute: int, memory: int, inbound_bytes: int,
                   energy: float) -> bool:
        if min(compute, memory, inbound_bytes) < 0 or not isfinite(energy) or energy < 0:
            return False
        u, b = self.usage, self.budget
        return (
            u.compute_units + compute <= b.compute_units
            and u.memory_items + memory <= b.memory_items
            and u.inbound_bytes + inbound_bytes <= b.inbound_bytes
            and u.energy_units + energy <= b.energy_units
        )

    def consume(self, *, compute: int = 0, memory: int = 0,
                inbound_bytes: int = 0, energy: float = 0.0) -> bool:
        """Consume all requested resources or none; never partially debit."""
        if not self.can_afford(compute=compute, memory=memory,
                               inbound_bytes=inbound_bytes, energy=energy):
            return False
        self.usage = ResourceUsage(
            self.usage.compute_units + compute,
            self.usage.memory_items + memory,
            self.usage.inbound_bytes + inbound_bytes,
            self.usage.energy_units + energy,
        )
        self._assert_invariants()
        return True

    def receive_and_validate(
        self,
        message: EvidenceMessage,
        local_observations: Iterable[Observation],
        *,
        required_accuracy: float = 0.9,
        min_samples: int = 8,
        validation_compute_per_sample: int = 1,
        validation_energy_per_sample: float = 0.0,
    ) -> ReceiveResult:
        """Receive a claim and test it against receiver-local observations.

        A successful result means only that the claim met the configured finite
        local test. It is not a proof of universality or truth outside that scope.
        Imported sender evidence is stored as provenance, not merged into the
        receiver's validation sample.
        """
        if not 0 <= required_accuracy <= 1 or not isfinite(required_accuracy):
            raise ValueError("required_accuracy must be in [0, 1]")
        if min_samples < 1 or validation_compute_per_sample < 0:
            raise ValueError("min_samples must be positive and compute cost non-negative")
        _finite_nonnegative(validation_energy_per_sample, "validation_energy_per_sample")
        size = message.encoded_size_bytes
        if message.message_id in self.messages:
            return ReceiveResult(False, "duplicate_message_id", message.message_id, size)
        if len(self.messages) >= self.budget.memory_items:
            return ReceiveResult(False, "memory_budget_exhausted", message.message_id, size)
        if size > self.budget.inbound_bytes - self.usage.inbound_bytes:
            return ReceiveResult(False, "inbound_byte_budget_exhausted", message.message_id, size)

        local = tuple(local_observations)
        if any(not isinstance(o, Observation) for o in local):
            raise TypeError("local_observations must contain Observation values")
        # Deterministic estimate of validation work; it is a policy unit, not CPU time.
        compute_cost = len(local) * validation_compute_per_sample
        energy_cost = len(local) * validation_energy_per_sample
        if not self.can_afford(compute=compute_cost, memory=1, inbound_bytes=size,
                               energy=energy_cost):
            return ReceiveResult(False, "resource_budget_exhausted", message.message_id, size)

        # Charge only after all preconditions have passed; validation is atomic.
        self.consume(compute=compute_cost, memory=1, inbound_bytes=size, energy=energy_cost)
        self.messages[message.message_id] = message
        digest_payload = [
            {"distance": o.distance, "relation_holds": o.relation_holds} for o in local
        ]
        digest = sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()
        if len(local) < min_samples:
            result = ValidationResult(message.message_id, self.cell_id, len(local), 0.0,
                                      False, required_accuracy, digest,
                                      "insufficient_independent_local_samples")
        else:
            correct = sum(
                (o.distance <= message.claim.threshold) == o.relation_holds for o in local
            )
            accuracy = correct / len(local)
            passed = accuracy >= required_accuracy
            result = ValidationResult(
                message.message_id, self.cell_id, len(local), accuracy, passed,
                required_accuracy, digest,
                "passed_finite_local_test" if passed else "failed_finite_local_test",
            )
        self.validations[message.message_id] = result
        self._assert_invariants()
        return ReceiveResult(True, "received_and_tested", message.message_id, size, result)


def make_threshold_message(
    *, message_id: str, origin_cell: str, sequence: int, threshold: float,
    scope: str, observations: Iterable[Observation], compute_cost: int = 1,
    energy_cost: float = 0.0, parent_message_ids: tuple[str, ...] = (),
) -> EvidenceMessage:
    """Validated constructor used by cells when publishing a threshold hypothesis."""
    return EvidenceMessage(
        message_id=message_id, schema_version=SCHEMA_VERSION, claim_kind=CLAIM_KIND,
        claim=ThresholdClaim(threshold, scope), observations=tuple(observations),
        origin_cell=origin_cell, created_sequence=sequence,
        declared_compute_cost=compute_cost, declared_energy_cost=energy_cost,
        parent_message_ids=parent_message_ids,
    )
