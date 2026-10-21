"""ECF v4.26: controlled benchmark for experience exchange.

The benchmark asks a narrow, falsifiable question: under a fixed number of
candidate validations per cell, does exchanging candidate relations improve
held-out performance over isolated cells? A synthetic environment supplies
binary feedback for a hidden distance-threshold relation. This feedback is an
evaluation oracle for the benchmark only; it is not claimed to be available in
real-world unsupervised discovery.

No neural networks are used. The candidate family and benchmark are explicit
human-designed inductive biases, not self-created mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence

import numpy as np

Condition = Literal["isolated", "neighbor_exchange", "randomized_exchange"]


@dataclass(frozen=True)
class ExperimentConfig:
    target_threshold: float = 0.37
    candidate_grid: tuple[float, ...] = tuple(i / 20 for i in range(1, 20))
    cells: int = 8
    candidates_per_cell: int = 4
    share_count: int = 2
    train_pairs: int = 96
    validation_pairs: int = 96
    test_pairs: int = 512

    def __post_init__(self) -> None:
        if not isfinite(self.target_threshold) or not 0 < self.target_threshold < 1:
            raise ValueError("target_threshold must be finite and in (0, 1)")
        if len(self.candidate_grid) < 2:
            raise ValueError("candidate_grid must contain at least two thresholds")
        if any(not isfinite(t) or not 0 < t < 1 for t in self.candidate_grid):
            raise ValueError("candidate thresholds must be finite and in (0, 1)")
        if len(set(self.candidate_grid)) != len(self.candidate_grid):
            raise ValueError("candidate thresholds must be unique")
        if self.cells < 4:
            raise ValueError("at least four cells are required")
        if not 1 <= self.candidates_per_cell <= len(self.candidate_grid):
            raise ValueError("candidates_per_cell must fit candidate_grid")
        if not 1 <= self.share_count <= self.candidates_per_cell:
            raise ValueError("share_count must be in [1, candidates_per_cell]")
        if min(self.train_pairs, self.validation_pairs, self.test_pairs) < 16:
            raise ValueError("each dataset split must contain at least 16 pairs")


@dataclass(frozen=True)
class CandidateEvidence:
    threshold: float
    discovery_accuracy: float
    origin_cell: int


@dataclass(frozen=True)
class CellOutcome:
    seed: int
    cell_id: int
    condition: str
    selected_threshold: float
    validation_accuracy: float
    test_accuracy: float
    threshold_error: float
    candidate_pool_size: int
    candidates_evaluated: int


@dataclass(frozen=True)
class ArmSummary:
    condition: str
    mean_test_accuracy: float
    mean_threshold_error: float
    mean_validation_accuracy: float
    mean_candidate_pool_size: float
    mean_candidates_evaluated: float
    observations: int


@dataclass(frozen=True)
class ExperimentReport:
    config: ExperimentConfig
    seeds: tuple[int, ...]
    summaries: tuple[ArmSummary, ...]
    outcomes: tuple[CellOutcome, ...]
    paired_accuracy_gain_vs_isolated: tuple[tuple[str, float], ...]
    interpretation: tuple[str, ...]


def _accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predicted == labels))


def _evaluate_threshold(threshold: float, distances: np.ndarray,
                        labels: np.ndarray) -> float:
    return _accuracy(distances <= threshold, labels)


def _make_data(rng: np.random.Generator, n: int,
               target: float) -> tuple[np.ndarray, np.ndarray]:
    distances = rng.random(n)
    labels = distances <= target
    return distances, labels


def _ring(cells: int) -> dict[int, tuple[int, ...]]:
    return {
        i: tuple(sorted({(i - 1) % cells, (i + 1) % cells}))
        for i in range(cells)
    }


def _unique_candidates(items: Sequence[CandidateEvidence]) -> list[CandidateEvidence]:
    best: dict[float, CandidateEvidence] = {}
    for item in items:
        previous = best.get(item.threshold)
        if previous is None or item.discovery_accuracy > previous.discovery_accuracy:
            best[item.threshold] = item
    return sorted(best.values(), key=lambda x: (-x.discovery_accuracy, x.threshold))


def run_exchange_experiment(
    seeds: Sequence[int] = tuple(range(20)),
    *,
    config: ExperimentConfig = ExperimentConfig(),
) -> ExperimentReport:
    """Run paired isolated, neighbor-sharing and randomized-sharing conditions.

    Every arm uses identical per-seed cell datasets and initial candidate
    proposals. Candidate evaluations per cell are capped at candidates_per_cell
    in every condition. Neighbors form a ring. Randomized exchange preserves
    the number of peer sources but samples them without using the ring edges.
    Selection uses validation data; final accuracy uses a separate test split.
    """
    seed_values = tuple(int(s) for s in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    if any(s < 0 for s in seed_values):
        raise ValueError("seeds must be non-negative")

    conditions: tuple[Condition, ...] = (
        "isolated", "neighbor_exchange", "randomized_exchange"
    )
    all_outcomes: list[CellOutcome] = []
    neighbors = _ring(config.cells)

    for seed in seed_values:
        # A single reproducible stream defines all data/proposals for this seed.
        rng = np.random.default_rng(seed)
        datasets = {}
        local_candidates: dict[int, list[CandidateEvidence]] = {}
        for cell_id in range(config.cells):
            train = _make_data(rng, config.train_pairs, config.target_threshold)
            valid = _make_data(rng, config.validation_pairs, config.target_threshold)
            test = _make_data(rng, config.test_pairs, config.target_threshold)
            datasets[cell_id] = (train, valid, test)
            proposed = rng.choice(
                np.asarray(config.candidate_grid),
                size=config.candidates_per_cell,
                replace=False,
            )
            train_distances, train_labels = train
            scored = [
                CandidateEvidence(
                    threshold=float(t),
                    discovery_accuracy=_evaluate_threshold(
                        float(t), train_distances, train_labels
                    ),
                    origin_cell=cell_id,
                )
                for t in proposed
            ]
            local_candidates[cell_id] = sorted(
                scored, key=lambda x: (-x.discovery_accuracy, x.threshold)
            )

        # Each condition reuses the same datasets and initial proposals.
        for condition in conditions:
            for cell_id in range(config.cells):
                own = local_candidates[cell_id]
                if condition == "isolated":
                    pool = list(own)
                else:
                    if condition == "neighbor_exchange":
                        peers = list(neighbors[cell_id])
                    else:
                        choices = [i for i in range(config.cells) if i != cell_id]
                        peers = list(rng.choice(
                            choices, size=len(neighbors[cell_id]), replace=False
                        ))
                    received = []
                    for peer in peers:
                        received.extend(local_candidates[peer][:config.share_count])
                    pool = _unique_candidates([*own, *received])

                # Fixed evaluation budget: sharing changes candidate availability,
                # not the number of candidates a cell is allowed to validate.
                pool = pool[:config.candidates_per_cell]
                train, valid, test = datasets[cell_id]
                valid_distances, valid_labels = valid
                test_distances, test_labels = test
                scored_validation = [
                    (
                        _evaluate_threshold(c.threshold, valid_distances, valid_labels),
                        c.threshold,
                    )
                    for c in pool
                ]
                # Deterministic tie-break: lower threshold.
                validation_accuracy, selected_threshold = max(
                    scored_validation, key=lambda x: (x[0], -x[1])
                )
                test_accuracy = _evaluate_threshold(
                    selected_threshold, test_distances, test_labels
                )
                all_outcomes.append(CellOutcome(
                    seed=seed,
                    cell_id=cell_id,
                    condition=condition,
                    selected_threshold=selected_threshold,
                    validation_accuracy=validation_accuracy,
                    test_accuracy=test_accuracy,
                    threshold_error=abs(selected_threshold - config.target_threshold),
                    candidate_pool_size=len(
                        own if condition == "isolated"
                        else _unique_candidates([*own, *received])
                    ),
                    candidates_evaluated=len(pool),
                ))

    summaries = []
    for condition in conditions:
        rows = [r for r in all_outcomes if r.condition == condition]
        summaries.append(ArmSummary(
            condition=condition,
            mean_test_accuracy=float(np.mean([r.test_accuracy for r in rows])),
            mean_threshold_error=float(np.mean([r.threshold_error for r in rows])),
            mean_validation_accuracy=float(np.mean([r.validation_accuracy for r in rows])),
            mean_candidate_pool_size=float(np.mean([r.candidate_pool_size for r in rows])),
            mean_candidates_evaluated=float(np.mean([r.candidates_evaluated for r in rows])),
            observations=len(rows),
        ))

    # Paired differences by seed and cell reduce variance in comparisons.
    isolated_by_key = {
        (r.seed, r.cell_id): r.test_accuracy
        for r in all_outcomes if r.condition == "isolated"
    }
    gains = []
    confidence_intervals = []
    for condition_index, condition in enumerate(conditions[1:]):
        rows = [r for r in all_outcomes if r.condition == condition]
        by_seed: dict[int, list[float]] = {}
        for row in rows:
            by_seed.setdefault(row.seed, []).append(
                row.test_accuracy - isolated_by_key[(row.seed, row.cell_id)]
            )
        # Bootstrap independent seeds as clusters, not individual cells, because
        # cells in a network are not statistically independent observations.
        seed_gains = np.asarray(
            [float(np.mean(by_seed[s])) for s in seed_values], dtype=float
        )
        gain = float(np.mean(seed_gains))
        gains.append((condition, gain))
        if len(seed_gains) < 2:
            confidence_intervals.append((condition, None, None))
        else:
            bootstrap_rng = np.random.default_rng(20261021 + condition_index)
            sampled = bootstrap_rng.choice(
                seed_gains, size=(4000, len(seed_gains)), replace=True
            )
            bootstrap_means = np.mean(sampled, axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            confidence_intervals.append((condition, float(low), float(high)))

    interpretation = (
        "Synthetic benchmark only: labels come from a hidden threshold oracle.",
        "Candidate generation is restricted to a human-specified threshold grid.",
        "A positive gain is evidence for this benchmark, not proof of general intelligence.",
        "Neighbor vs randomized exchange tests whether ring topology adds value beyond pooling.",
        "Use more independent seeds and report uncertainty before drawing strong conclusions.",
    )
    return ExperimentReport(
        config=config,
        seeds=seed_values,
        summaries=tuple(summaries),
        outcomes=tuple(all_outcomes),
        paired_accuracy_gain_vs_isolated=tuple(gains),
        interpretation=interpretation,
    )
