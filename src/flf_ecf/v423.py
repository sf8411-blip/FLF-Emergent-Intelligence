"""ECF v4.23: counterexample-guided hypothesis assessment and revision.

A candidate can be refuted by a witness or survive bounded finite searches.
Survival is evidence, never a proof. Revision is supplied by a caller so this
module does not silently invent domain-specific repair rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .v419 import Expr
from .v420 import ArrayTransform
from .v422 import Counterexample, CounterexampleReport, search_counterexamples


@dataclass(frozen=True)
class HypothesisAssessment:
    expression: Expr
    status: str
    discovery: CounterexampleReport
    validation: CounterexampleReport

    @property
    def counterexamples(self) -> tuple[Counterexample, ...]:
        return self.discovery.counterexamples + self.validation.counterexamples


@dataclass(frozen=True)
class RevisionRecord:
    revision_index: int
    assessment: HypothesisAssessment


@dataclass(frozen=True)
class RevisionRun:
    records: tuple[RevisionRecord, ...]
    final_expression: Expr
    status: str


def assess_hypothesis(
    expr: Expr,
    transform: ArrayTransform,
    lower: float,
    upper: float,
    *,
    n_samples: int = 1000,
    tolerance: float = 1e-8,
    seed: int = 0,
) -> HypothesisAssessment:
    """Run discovery and independent-seed validation searches on a bounded box."""
    discovery = search_counterexamples(
        expr, transform, lower, upper, n_samples=n_samples,
        tolerance=tolerance, seed=seed,
    )
    validation = search_counterexamples(
        expr, transform, lower, upper, n_samples=n_samples,
        tolerance=tolerance, seed=seed + 1,
    )
    if discovery.refuted:
        status = "refuted_discovery"
    elif validation.refuted:
        status = "refuted_validation"
    else:
        status = "survived_finite_search"
    return HypothesisAssessment(expr, status, discovery, validation)


RevisionFunction = Callable[[Expr, tuple[Counterexample, ...]], Expr | None]


def run_revision_loop(
    initial: Expr,
    transform: ArrayTransform,
    lower: float,
    upper: float,
    revise: RevisionFunction,
    *,
    max_revisions: int = 3,
    n_samples: int = 1000,
    tolerance: float = 1e-8,
    seed: int = 0,
) -> RevisionRun:
    """Bounded counterexample-guided loop; never labels a hypothesis 'proved'.

    revise receives the current expression and all witnesses found so far.
    Return None to stop. Each candidate is reassessed from scratch using new,
    deterministic seeds. A surviving candidate means only that these finite
    searches did not find a counterexample.
    """
    if max_revisions < 0:
        raise ValueError("max_revisions must be non-negative")
    current = initial
    records: list[RevisionRecord] = []
    for index in range(max_revisions + 1):
        assessment = assess_hypothesis(
            current, transform, lower, upper,
            n_samples=n_samples, tolerance=tolerance, seed=seed + 2 * index,
        )
        records.append(RevisionRecord(index, assessment))
        if assessment.status == "survived_finite_search":
            return RevisionRun(tuple(records), current, "survived_finite_search")
        if index == max_revisions:
            break
        candidate = revise(current, assessment.counterexamples)
        if candidate is None:
            return RevisionRun(tuple(records), current, "revision_stopped")
        if not isinstance(candidate, Expr):
            raise TypeError("revision callback must return Expr or None")
        if candidate == current:
            return RevisionRun(tuple(records), current, "revision_no_change")
        current = candidate
    return RevisionRun(tuple(records), current, "revision_budget_exhausted")
