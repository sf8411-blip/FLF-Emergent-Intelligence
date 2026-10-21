"""ECF v4.24: minimal non-neural digital-cell substrate.

This is an experimental substrate for local state, bounded memory, and cautious
experience exchange. It is not an AGI implementation and does not infer truth
from message counts. Imported experiences remain unverified until retested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class CellGenome:
    """Structural/resource traits; intentionally not a neural-network weight set."""
    memory_limit: int = 128
    compute_budget: int = 100
    energy_budget: float = 100.0
    mutation_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.memory_limit < 1 or self.compute_budget < 1:
            raise ValueError("memory_limit and compute_budget must be positive")
        if not isfinite(self.energy_budget) or self.energy_budget <= 0:
            raise ValueError("energy_budget must be finite and positive")
        if not isfinite(self.mutation_rate) or not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")


@dataclass(frozen=True)
class Experience:
    """A claim plus its context and evidence; receipt is not endorsement."""
    experience_id: str
    claim: str
    context: str
    outcome: str  # observed_positive, observed_negative, or unresolved
    evidence: str
    origin_cell: str
    compute_cost: int = 1
    energy_cost: float = 0.0
    validated_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.experience_id, self.claim, self.context,
                    self.evidence, self.origin_cell)):
            raise ValueError("experience identity, claim, context, evidence and origin are required")
        if self.outcome not in {"observed_positive", "observed_negative", "unresolved"}:
            raise ValueError("unsupported experience outcome")
        if self.compute_cost < 0 or not isfinite(self.energy_cost) or self.energy_cost < 0:
            raise ValueError("experience costs must be non-negative and finite")

    @property
    def verified(self) -> bool:
        return len(set(self.validated_by)) >= 2


@dataclass
class DigitalCell:
    cell_id: str
    genome: CellGenome = field(default_factory=CellGenome)
    state: tuple[float, ...] = ()
    neighbors: set[str] = field(default_factory=set)
    memory: dict[str, Experience] = field(default_factory=dict)
    compute_used: int = 0
    energy_used: float = 0.0

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id is required")
        if not all(isfinite(x) for x in self.state):
            raise ValueError("cell state must contain finite values")

    @property
    def compute_remaining(self) -> int:
        return max(0, self.genome.compute_budget - self.compute_used)

    @property
    def energy_remaining(self) -> float:
        return max(0.0, self.genome.energy_budget - self.energy_used)

    def can_afford(self, compute_cost: int, energy_cost: float) -> bool:
        return (compute_cost >= 0 and isfinite(energy_cost) and energy_cost >= 0
                and compute_cost <= self.compute_remaining
                and energy_cost <= self.energy_remaining)

    def spend(self, compute_cost: int, energy_cost: float) -> bool:
        """Atomically consume budget, returning False if the host budget is insufficient."""
        if not self.can_afford(compute_cost, energy_cost):
            return False
        self.compute_used += compute_cost
        self.energy_used += energy_cost
        return True

    def remember(self, experience: Experience, *, imported: bool = False) -> bool:
        """Store within a strict bound; imported items are never auto-validated."""
        if experience.experience_id in self.memory:
            return False
        if not self.spend(experience.compute_cost, experience.energy_cost):
            return False
        if len(self.memory) >= self.genome.memory_limit:
            # Deterministic conservative eviction: discard the oldest insertion.
            oldest = next(iter(self.memory))
            del self.memory[oldest]
        if imported and experience.validated_by:
            experience = Experience(
                experience_id=experience.experience_id,
                claim=experience.claim,
                context=experience.context,
                outcome=experience.outcome,
                evidence=experience.evidence,
                origin_cell=experience.origin_cell,
                compute_cost=experience.compute_cost,
                energy_cost=experience.energy_cost,
                validated_by=(),
            )
        self.memory[experience.experience_id] = experience
        return True

    def export_experiences(self, limit: int = 16) -> tuple[Experience, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(list(self.memory.values())[:limit])


@dataclass(frozen=True)
class ExchangeReport:
    sender: str
    receiver: str
    offered: int
    accepted: int
    rejected: int


class CellNetwork:
    """Explicit local graph; exchange only occurs across declared neighbor links."""
    def __init__(self, cells: Iterable[DigitalCell] = ()) -> None:
        self.cells: dict[str, DigitalCell] = {}
        for cell in cells:
            self.add_cell(cell)

    def add_cell(self, cell: DigitalCell) -> None:
        if cell.cell_id in self.cells:
            raise ValueError("duplicate cell_id")
        self.cells[cell.cell_id] = cell

    def connect(self, left: str, right: str) -> None:
        if left == right or left not in self.cells or right not in self.cells:
            raise ValueError("connection requires two distinct known cells")
        self.cells[left].neighbors.add(right)
        self.cells[right].neighbors.add(left)

    def exchange(self, sender: str, receiver: str, *, limit: int = 16) -> ExchangeReport:
        if sender not in self.cells or receiver not in self.cells:
            raise ValueError("unknown sender or receiver")
        source, target = self.cells[sender], self.cells[receiver]
        if receiver not in source.neighbors or sender not in target.neighbors:
            raise ValueError("experience exchange requires an explicit neighbor link")
        offered = source.export_experiences(limit)
        accepted = 0
        for item in offered:
            if target.remember(item, imported=True):
                accepted += 1
        return ExchangeReport(sender, receiver, len(offered), accepted, len(offered) - accepted)
