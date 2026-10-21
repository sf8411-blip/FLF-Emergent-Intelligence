"""ECF v4.28: controlled exchange using typed messages and receiver validation.

Synthetic benchmark only. It uses a known threshold oracle to score candidates,
while every candidate admitted to a receiver's selection pool is revalidated
against that receiver's own validation observations through v4.27's ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence

import numpy as np

from .v427 import (
    CellLedger, EvidenceMessage, Observation, ResourceBudget,
    make_threshold_message,
)

Condition = Literal["isolated", "neighbor_exchange", "randomized_exchange"]
Scenario = Literal["clean", "false_claims", "correlated_evidence", "distribution_shift"]
CONDITIONS: tuple[Condition, ...] = (
    "isolated", "neighbor_exchange", "randomized_exchange"
)
SCENARIOS: tuple[Scenario, ...] = (
    "clean", "false_claims", "correlated_evidence", "distribution_shift"
)


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    target_threshold: float = 0.37
    shifted_test_threshold: float = 0.68
    candidate_grid: tuple[float, ...] = tuple(i / 20 for i in range(1, 20))
    cells: int = 8
    candidates_per_cell: int = 4
    share_count: int = 2
    train_pairs: int = 96
    validation_pairs: int = 96
    test_pairs: int = 512
    required_accuracy: float = 0.90
    min_validation_samples: int = 16
    validation_compute_per_sample: int = 1
    validation_energy_per_sample: float = 0.0
    false_claim_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not isfinite(self.target_threshold) or not 0 < self.target_threshold < 1:
            raise ValueError("target_threshold must be in (0, 1)")
        if not isfinite(self.shifted_test_threshold) or not 0 < self.shifted_test_threshold < 1:
            raise ValueError("shifted_test_threshold must be in (0, 1)")
        if len(self.candidate_grid) < 2 or len(set(self.candidate_grid)) != len(self.candidate_grid):
            raise ValueError("candidate_grid needs at least two unique thresholds")
        if any(not isfinite(t) or not 0 < t < 1 for t in self.candidate_grid):
            raise ValueError("candidate thresholds must be finite and in (0, 1)")
        if self.cells < 4:
            raise ValueError("at least four cells are required")
        if not 1 <= self.candidates_per_cell <= len(self.candidate_grid):
            raise ValueError("candidate budget must fit candidate_grid")
        if not 1 <= self.share_count <= self.candidates_per_cell:
            raise ValueError("share_count must be in [1, candidates_per_cell]")
        if min(self.train_pairs, self.validation_pairs, self.test_pairs) < 16:
            raise ValueError("all sample counts must be at least 16")
        if not 0 <= self.required_accuracy <= 1:
            raise ValueError("required_accuracy must be in [0, 1]")
        if self.min_validation_samples < 1 or self.validation_compute_per_sample < 0:
            raise ValueError("invalid validation sample/cost configuration")
        if not isfinite(self.validation_energy_per_sample) or self.validation_energy_per_sample < 0:
            raise ValueError("validation_energy_per_sample must be non-negative and finite")
        if not 0 <= self.false_claim_fraction <= 1:
            raise ValueError("false_claim_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CellTrial:
    seed: int
    scenario: str
    condition: str
    cell_id: int
    selected_threshold: float | None
    validation_accuracy: float
    test_accuracy: float
    threshold_error: float
    candidates_considered: int
    candidates_passed: int
    resource_compute_used: int
    resource_bytes_used: int
    rejected_false_candidates: int


@dataclass(frozen=True, slots=True)
class ArmSummary:
    scenario: str
    condition: str
    mean_test_accuracy: float
    selection_coverage: float
    mean_test_accuracy_when_selected: float | None
    mean_validation_accuracy: float
    mean_threshold_error: float
    mean_candidates_passed: float
    mean_rejected_false_candidates: float
    mean_compute_used: float
    mean_bytes_used: float
    observations: int


@dataclass(frozen=True, slots=True)
class ExchangeReport:
    config: ExchangeConfig
    seeds: tuple[int, ...]
    trials: tuple[CellTrial, ...]
    summaries: tuple[ArmSummary, ...]
    interpretation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Proposal:
    message: EvidenceMessage
    ranking_score: float
    forged: bool = False


def _data(rng: np.random.Generator, n: int, threshold: float):
    x = rng.random(n)
    return x, x <= threshold


def _observations(x: np.ndarray, y: np.ndarray) -> tuple[Observation, ...]:
    return tuple(Observation(float(a), bool(b)) for a, b in zip(x, y))


def _score(threshold: float, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x <= threshold) == y))


def _ring(cells: int) -> dict[int, tuple[int, ...]]:
    return {i: tuple(sorted({(i - 1) % cells, (i + 1) % cells})) for i in range(cells)}


def _message_size_bound(config: ExchangeConfig) -> int:
    # Conservative enough for the configured scalar observations and JSON fields.
    return 512 + config.train_pairs * 64


def run_typed_exchange_experiment(
    seeds: Sequence[int] = tuple(range(20)),
    *,
    config: ExchangeConfig = ExchangeConfig(),
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> ExchangeReport:
    """Compare isolated, ring-neighbor, and random-peer exchange.

    All conditions have the same maximum number of candidate validations per
    receiver. A receiver admits only messages that pass its own local validation.
    Scenario 'false_claims' injects forged sender evidence into a subset of
    candidates. 'correlated_evidence' reuses one sender training set across cells.
    'distribution_shift' changes the test-world threshold after validation.
    """
    seed_values = tuple(int(s) for s in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values) or min(seed_values) < 0:
        raise ValueError("seeds must be unique, non-empty, and non-negative")
    scenario_values = tuple(scenarios)
    if not scenario_values or any(s not in SCENARIOS for s in scenario_values):
        raise ValueError("unsupported scenario")
    all_trials: list[CellTrial] = []
    neighbors = _ring(config.cells)

    for seed in seed_values:
        for scenario_index, scenario in enumerate(scenario_values):
            rng = np.random.default_rng(seed * 1009 + scenario_index * 9176)
            data_by_cell = {}
            proposals: dict[int, list[_Proposal]] = {}
            shared_train = _data(rng, config.train_pairs, config.target_threshold)
            for cell_id in range(config.cells):
                train = shared_train if scenario == "correlated_evidence" else _data(
                    rng, config.train_pairs, config.target_threshold
                )
                valid = _data(rng, config.validation_pairs, config.target_threshold)
                test_target = (config.shifted_test_threshold
                               if scenario == "distribution_shift"
                               else config.target_threshold)
                test = _data(rng, config.test_pairs, test_target)
                data_by_cell[cell_id] = (train, valid, test)
                tx, ty = train
                chosen = rng.choice(np.asarray(config.candidate_grid),
                                    size=config.candidates_per_cell, replace=False)
                cell_proposals = []
                for j, raw_threshold in enumerate(chosen):
                    threshold = float(raw_threshold)
                    forged = False
                    if scenario == "false_claims" and (
                        (cell_id * config.candidates_per_cell + j)
                        < round(config.cells * config.candidates_per_cell
                                * config.false_claim_fraction)
                    ):
                        # Deliberately false claim plus fabricated sender observations
                        # that make that claim look perfect. Receiver must not trust them.
                        threshold = 0.95 if config.target_threshold < 0.7 else 0.05
                        fake_x = np.linspace(0.001, 0.999, config.train_pairs)
                        fake_y = fake_x <= threshold
                        evidence = _observations(fake_x, fake_y)
                        ranking_score = 1.0
                        forged = True
                    else:
                        evidence = _observations(tx, ty)
                        ranking_score = _score(threshold, tx, ty)
                    msg = make_threshold_message(
                        message_id=f"s{seed}-sc{scenario}-c{cell_id}-p{j}",
                        origin_cell=f"cell-{cell_id}", sequence=seed,
                        threshold=threshold, scope="synthetic-distance-threshold-world",
                        observations=evidence,
                        compute_cost=config.train_pairs,
                    )
                    cell_proposals.append(_Proposal(msg, ranking_score, forged))
                proposals[cell_id] = sorted(
                    cell_proposals,
                    key=lambda p: (-p.ranking_score, p.message.claim.threshold,
                                   p.message.message_id),
                )

            for condition in CONDITIONS:
                for cell_id in range(config.cells):
                    own = proposals[cell_id]
                    if condition == "isolated":
                        pool = list(own)
                    else:
                        if condition == "neighbor_exchange":
                            peers = neighbors[cell_id]
                        else:
                            options = [i for i in range(config.cells) if i != cell_id]
                            peers = tuple(int(i) for i in rng.choice(
                                options, size=len(neighbors[cell_id]), replace=False
                            ))
                        received = [p for peer in peers for p in proposals[peer][:config.share_count]]
                        # Deduplicate equivalent thresholds, retaining best sender evidence rank.
                        best: dict[float, _Proposal] = {p.message.claim.threshold: p for p in own}
                        for p in received:
                            t = p.message.claim.threshold
                            if t not in best or p.ranking_score > best[t].ranking_score:
                                best[t] = p
                        pool = sorted(best.values(), key=lambda p: (
                            -p.ranking_score, p.message.claim.threshold, p.message.message_id
                        ))
                    pool = pool[:config.candidates_per_cell]

                    vx, vy = data_by_cell[cell_id][1]
                    local_validation = _observations(vx, vy)
                    max_message_bytes = _message_size_bound(config)
                    ledger = CellLedger(
                        f"cell-{cell_id}",
                        ResourceBudget(
                            compute_units=(config.candidates_per_cell
                                           * config.validation_pairs
                                           * config.validation_compute_per_sample),
                            memory_items=config.candidates_per_cell,
                            inbound_bytes=config.candidates_per_cell * max_message_bytes,
                            energy_units=(config.candidates_per_cell
                                          * config.validation_pairs
                                          * config.validation_energy_per_sample),
                        ),
                    )
                    accepted = []
                    false_rejected = 0
                    considered = 0
                    for proposal in pool:
                        considered += 1
                        result = ledger.receive_and_validate(
                            proposal.message, local_validation,
                            required_accuracy=config.required_accuracy,
                            min_samples=config.min_validation_samples,
                            validation_compute_per_sample=config.validation_compute_per_sample,
                            validation_energy_per_sample=config.validation_energy_per_sample,
                        )
                        if result.accepted and result.validation and result.validation.passed:
                            accepted.append((result.validation.accuracy,
                                             proposal.message.claim.threshold, proposal))
                        elif proposal.forged:
                            false_rejected += 1
                    tx, ty = data_by_cell[cell_id][0]
                    _, test_y = data_by_cell[cell_id][2]
                    test_x, _ = data_by_cell[cell_id][2]
                    if accepted:
                        # Highest local validation accuracy, deterministic threshold tie-break.
                        va, selected, _ = max(accepted, key=lambda item: (item[0], -item[1]))
                        test_acc = _score(selected, test_x, test_y)
                        err = abs(selected - (
                            config.shifted_test_threshold if scenario == "distribution_shift"
                            else config.target_threshold
                        ))
                    else:
                        va, selected, test_acc, err = 0.0, None, 0.0, 1.0
                    all_trials.append(CellTrial(
                        seed=seed, scenario=scenario, condition=condition, cell_id=cell_id,
                        selected_threshold=selected, validation_accuracy=va,
                        test_accuracy=test_acc, threshold_error=err,
                        candidates_considered=considered,
                        candidates_passed=len(accepted),
                        resource_compute_used=ledger.usage.compute_units,
                        resource_bytes_used=ledger.usage.inbound_bytes,
                        rejected_false_candidates=false_rejected,
                    ))

    summaries = []
    for scenario in scenario_values:
        for condition in CONDITIONS:
            rows = [t for t in all_trials
                    if t.scenario == scenario and t.condition == condition]
            summaries.append(ArmSummary(
                scenario=scenario, condition=condition,
                mean_test_accuracy=float(np.mean([r.test_accuracy for r in rows])),
                selection_coverage=float(np.mean([r.selected_threshold is not None for r in rows])),
                mean_test_accuracy_when_selected=(
                    float(np.mean([r.test_accuracy for r in rows if r.selected_threshold is not None]))
                    if any(r.selected_threshold is not None for r in rows) else None
                ),
                mean_validation_accuracy=float(np.mean([r.validation_accuracy for r in rows])),
                mean_threshold_error=float(np.mean([r.threshold_error for r in rows])),
                mean_candidates_passed=float(np.mean([r.candidates_passed for r in rows])),
                mean_rejected_false_candidates=float(np.mean([r.rejected_false_candidates for r in rows])),
                mean_compute_used=float(np.mean([r.resource_compute_used for r in rows])),
                mean_bytes_used=float(np.mean([r.resource_bytes_used for r in rows])),
                observations=len(rows),
            ))
    return ExchangeReport(
        config=config, seeds=seed_values, trials=tuple(all_trials), summaries=tuple(summaries),
        interpretation=(
            "Synthetic threshold-oracle benchmark only; not evidence of AGI or autonomous mathematics.",
            "Receiver validation uses receiver-local labels; sender evidence is provenance, not receiver validation.",
            "False-claim injection tests whether local revalidation rejects fabricated sender evidence.",
            "Correlated-evidence scenario reuses sender training observations; receiver validation remains local.",
            "Distribution shift changes the held-out test threshold after validation and can invalidate previously useful claims.",
            "Resource counters are deterministic policy units, not measured CPU time or physical joules.",
        ),
    )
