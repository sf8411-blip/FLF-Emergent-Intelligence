"""ECF v4.25: auditable relational discovery from unlabeled observations.

This is a complete, bounded experimental subsystem, not an AGI claim. It uses
explicit geometric primitives as an initial inductive bias and exposes that
bias in every report. It does not use neural networks or train on human labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence
import numpy as np


@dataclass(frozen=True)
class Observation:
    observation_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must not be empty")
        if not self.values or not all(isfinite(v) for v in self.values):
            raise ValueError("values must be a non-empty finite vector")


@dataclass(frozen=True)
class RelationHypothesis:
    relation_id: str
    kind: str
    threshold: float
    dimension: int
    description: str


@dataclass(frozen=True)
class PairEvidence:
    left_id: str
    right_id: str
    distance: float
    predicted_relation: str


@dataclass(frozen=True)
class RelationAssessment:
    hypothesis: RelationHypothesis
    train_pairs: int
    validation_pairs: int
    train_positive_rate: float
    validation_positive_rate: float
    train_validation_gap: float
    stable: bool
    evidence: tuple[PairEvidence, ...]
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryReport:
    observations: int
    feature_dimension: int
    hypotheses: tuple[RelationAssessment, ...]
    rejected_reason: str | None = None


def _matrix(observations: Sequence[Observation]) -> np.ndarray:
    if len(observations) < 4:
        raise ValueError("at least four observations are required")
    ids = [o.observation_id for o in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("observation IDs must be unique")
    dimension = len(observations[0].values)
    if any(len(o.values) != dimension for o in observations):
        raise ValueError("all observations must have the same feature dimension")
    return np.asarray([o.values for o in observations], dtype=float)


def discover_distance_relations(
    observations: Iterable[Observation],
    *,
    thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    validation_fraction: float = 0.3,
    min_pair_support: int = 4,
    stability_tolerance: float = 0.25,
    max_evidence_pairs: int = 32,
    seed: int = 0,
) -> DiscoveryReport:
    """Propose proximity relations and test their support on held-out pairs.

    Values are standardized using discovery observations only. Pair splits are
    deterministic for a fixed seed. A relation is called stable only when the
    positive-pair rate differs by no more than stability_tolerance between the
    discovery and held-out pair sets and both sets meet min_pair_support.

    Important: this searches only one explicit family of Euclidean proximity
    relations. Stability is empirical support, not truth or mathematical proof.
    """
    items = tuple(observations)
    X = _matrix(items)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if min_pair_support < 1 or max_evidence_pairs < 0:
        raise ValueError("support must be positive and evidence limit non-negative")
    if not isfinite(stability_tolerance) or not 0 <= stability_tolerance <= 1:
        raise ValueError("stability_tolerance must be in [0, 1]")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if any(not isfinite(float(t)) or float(t) <= 0 for t in thresholds):
        raise ValueError("thresholds must be finite and positive")
    if len(set(float(t) for t in thresholds)) != len(thresholds):
        raise ValueError("thresholds must be unique")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(items))
    split = max(2, min(len(items) - 2, int(round(len(items) * (1 - validation_fraction)))))
    train_idx, valid_idx = permutation[:split], permutation[split:]
    # Fit normalization on discovery observations only to avoid validation leakage.
    mean = X[train_idx].mean(axis=0)
    scale = X[train_idx].std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    Z = (X - mean) / scale

    def pair_rows(indices: np.ndarray) -> list[tuple[int, int, float]]:
        rows = []
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                distance = float(np.linalg.norm(Z[i] - Z[j]) / np.sqrt(Z.shape[1]))
                rows.append((int(i), int(j), distance))
        return rows

    train_pairs = pair_rows(train_idx)
    valid_pairs = pair_rows(valid_idx)
    assessments = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        train_positive = sum(d <= threshold for _, _, d in train_pairs)
        valid_positive = sum(d <= threshold for _, _, d in valid_pairs)
        train_rate = train_positive / len(train_pairs) if train_pairs else 0.0
        valid_rate = valid_positive / len(valid_pairs) if valid_pairs else 0.0
        gap = abs(train_rate - valid_rate)
        enough = train_positive >= min_pair_support and valid_positive >= min_pair_support
        stable = enough and gap <= stability_tolerance
        relation = RelationHypothesis(
            relation_id=f"euclidean_proximity_{threshold:g}",
            kind="proximity",
            threshold=threshold,
            dimension=Z.shape[1],
            description=(
                "Two observations are related when their discovery-normalized "
                "Euclidean distance divided by sqrt(feature_dimension) is "
                f"<= {threshold:g}."
            ),
        )
        combined = sorted(train_pairs + valid_pairs, key=lambda row: (row[2], row[0], row[1]))
        evidence = tuple(
            PairEvidence(items[i].observation_id, items[j].observation_id, d,
                         "related" if d <= threshold else "not_related")
            for i, j, d in combined[:max_evidence_pairs]
        )
        caveats = [
            "The candidate family is human-specified Euclidean proximity.",
            "Finite pair support does not establish a universal relation.",
            "Feature scaling changes the geometry and is part of the hypothesis.",
        ]
        if not enough:
            caveats.append("Insufficient positive-pair support in train or validation.")
        if gap > stability_tolerance:
            caveats.append("Discovery and validation positive-pair rates disagree.")
        assessments.append(RelationAssessment(
            hypothesis=relation,
            train_pairs=len(train_pairs),
            validation_pairs=len(valid_pairs),
            train_positive_rate=train_rate,
            validation_positive_rate=valid_rate,
            train_validation_gap=gap,
            stable=stable,
            evidence=evidence,
            caveats=tuple(caveats),
        ))
    assessments.sort(key=lambda a: (
        not a.stable, a.train_validation_gap,
        -min(a.train_positive_rate, a.validation_positive_rate),
        a.hypothesis.threshold,
    ))
    return DiscoveryReport(
        observations=len(items),
        feature_dimension=Z.shape[1],
        hypotheses=tuple(assessments),
    )
