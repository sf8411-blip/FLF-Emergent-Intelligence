"""ECF v4.29: controlled comparison of evidence-guided structure-discovery strategies.

This is a synthetic, bounded benchmark. It compares candidate-generation policies,
not AGI. Every strategy uses the same typed expression grammar and evaluation budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal
import numpy as np

from .v419 import Expr, evaluate, variable

Strategy = Literal["enumeration", "random_search", "evolutionary", "hierarchical", "evidence_guided"]
STRATEGIES: tuple[Strategy, ...] = (
    "enumeration", "random_search", "evolutionary", "hierarchical", "evidence_guided"
)


@dataclass(frozen=True)
class DiscoveryConfig:
    seeds: int = 50
    samples: int = 384
    candidate_budget: int = 40
    validation_fraction: float = 0.25
    test_fraction: float = 0.25
    noise_std: float = 0.03
    max_complexity: int = 5

    def __post_init__(self):
        if self.seeds < 1 or self.samples < 64 or self.candidate_budget < 1:
            raise ValueError("seeds, samples, and candidate_budget must be positive (samples >= 64)")
        if not 0 < self.validation_fraction < 0.5 or not 0 < self.test_fraction < 0.5:
            raise ValueError("validation/test fractions must be in (0, .5)")
        if self.validation_fraction + self.test_fraction >= 0.8:
            raise ValueError("validation and test fractions leave too little training data")
        if self.noise_std < 0 or not np.isfinite(self.noise_std):
            raise ValueError("noise_std must be finite and non-negative")
        if self.max_complexity < 2:
            raise ValueError("max_complexity must be >= 2")


@dataclass(frozen=True)
class DiscoveryTrial:
    seed: int
    strategy: str
    test_rmse: float
    baseline_test_rmse: float
    relative_gain: float
    target_components_found: int
    candidates_evaluated: int
    selected_complexity: int
    validation_rmse: float


@dataclass(frozen=True)
class DiscoverySummary:
    strategy: str
    mean_test_rmse: float
    mean_relative_gain: float
    component_recovery_rate: float
    mean_candidates_evaluated: float
    mean_selected_complexity: float
    trials: int


@dataclass(frozen=True)
class DiscoveryReport:
    config: DiscoveryConfig
    trials: tuple[DiscoveryTrial, ...]
    summaries: tuple[DiscoverySummary, ...]
    notes: tuple[str, ...]


def _expr_key(e: Expr) -> tuple:
    return (e.op, e.index, tuple(_expr_key(a) for a in e.args))


def _pool() -> tuple[Expr, ...]:
    v = (variable(0), variable(1), variable(2))
    out: list[Expr] = []
    for a in v:
        out.extend((Expr("square", (a,)), Expr("sin", (a,)), Expr("abs", (a,))))
    for a, b in combinations(v, 2):
        out.extend((Expr("add", (a, b)), Expr("mul", (a, b))))
    # Include one small nested family to test structural composition.
    out.extend(Expr("square", (a,)) for a in v)
    return tuple(dict.fromkeys(out))


def _evolve_expr(parent: Expr, rng: np.random.Generator) -> Expr:
    vars_ = [variable(i) for i in range(3)]
    unary = ("square", "sin", "abs")
    if rng.random() < 0.45:
        return Expr(str(rng.choice(unary)), (vars_[int(rng.integers(0, 3))],))
    other = vars_[int(rng.integers(0, 3))]
    op = str(rng.choice(("add", "mul")))
    return Expr(op, (parent, other))


def _candidate_order(strategy: Strategy, rng: np.random.Generator,
                     budget: int, X: np.ndarray, y: np.ndarray) -> tuple[Expr, ...]:
    pool = list(_pool())
    if strategy == "enumeration":
        ordered = sorted(pool, key=lambda e: (e.complexity, _expr_key(e)))
    elif strategy == "hierarchical":
        # Tree-inspired hierarchy: simple unary operators first, then pairwise
        # compositions, then deeper expressions. It is a search order, not a tree model.
        ordered = sorted(pool, key=lambda e: (
            0 if e.op in {"square", "sin", "abs"} and e.args[0].op == "var" else 1,
            e.complexity, _expr_key(e)))
    elif strategy == "random_search":
        ordered = [pool[i] for i in rng.permutation(len(pool))]
        while len(ordered) < budget:
            ordered.append(_evolve_expr(ordered[int(rng.integers(len(ordered)))], rng))
    elif strategy == "evidence_guided":
        # Rank one-operator candidates by reduction in training residual SSE.
        base = np.column_stack((np.ones(len(X)), X))
        base_coef, *_ = np.linalg.lstsq(base, y, rcond=None)
        residual = y - base @ base_coef
        def relevance(e: Expr) -> tuple:
            try:
                z = evaluate(e, X)
                z = z - z.mean()
                denom = float(np.linalg.norm(z) * np.linalg.norm(residual))
                corr = abs(float(z @ residual) / denom) if denom > 1e-12 else 0.0
                return (-corr, e.complexity, _expr_key(e))
            except (ValueError, FloatingPointError):
                return (0.0, e.complexity, _expr_key(e))
        ordered = sorted(pool, key=relevance)
    else:  # evolutionary: retain and mutate candidates with lowest training RMSE
        ordered = []
        seen = set()
        population = [pool[int(i)] for i in rng.choice(len(pool), size=min(8, len(pool)), replace=False)]
        base = np.column_stack((np.ones(len(X)), X))
        for _ in range(max(budget * 3, 1)):
            if not population:
                break
            scored = []
            for e in population:
                if e.complexity > 5:
                    continue
                try:
                    z = evaluate(e, X)
                    design = np.column_stack((base, z))
                    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
                    score = float(np.mean((y - design @ coef) ** 2))
                    scored.append((score, e))
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    continue
            if not scored:
                population = [pool[int(rng.integers(len(pool)))]]
                continue
            scored.sort(key=lambda pair: (pair[0], _expr_key(pair[1])))
            for _, e in scored:
                key = _expr_key(e)
                if key not in seen:
                    ordered.append(e)
                    seen.add(key)
                    if len(ordered) >= budget:
                        break
            if len(ordered) >= budget:
                break
            parents = [e for _, e in scored[:max(1, len(scored)//2)]]
            population = parents + [_evolve_expr(parents[int(rng.integers(len(parents)))], rng)
                                    for _ in range(min(8, len(parents) + 2))]
        ordered.extend(e for e in pool if _expr_key(e) not in seen)
    # All methods are subject to the same candidate-evaluation cap.
    unique = []
    keys = set()
    for e in ordered:
        k = _expr_key(e)
        if k not in keys and e.complexity <= 5:
            unique.append(e)
            keys.add(k)
        if len(unique) >= budget:
            break
    # Keep the unique-proposal budget equal across strategies even when a
    # simple-first ordering exhausts the shallow grammar. Fill only with safe,
    # bounded compositions; strategy-specific ordering still determines priority.
    attempts = 0
    while len(unique) < budget and attempts < budget * 50:
        attempts += 1
        parent = unique[int(rng.integers(len(unique)))] if unique else pool[0]
        candidate = _evolve_expr(parent, rng)
        key = _expr_key(candidate)
        if candidate.complexity <= 5 and key not in keys:
            unique.append(candidate)
            keys.add(key)
    return tuple(unique)


def _fit_score(X: np.ndarray, y: np.ndarray, terms: tuple[Expr, ...]) -> float:
    cols = [np.ones(len(X))]
    cols.extend(X[:, i] for i in range(X.shape[1]))
    cols.extend(evaluate(e, X) for e in terms)
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(np.sqrt(np.mean((y - design @ coef) ** 2)))


def _greedy_select(X: np.ndarray, y: np.ndarray, candidates: tuple[Expr, ...],
                   max_terms: int = 2) -> tuple[tuple[Expr, ...], float, int]:
    selected: tuple[Expr, ...] = ()
    remaining = list(candidates)
    evaluations = 0
    for _ in range(max_terms):
        if not remaining:
            break
        scored = []
        for e in remaining:
            if e in selected:
                continue
            evaluations += 1
            try:
                score = _fit_score(X, y, selected + (e,))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                score = float("inf")
            scored.append((score, e))
        if not scored:
            break
        score, best = min(scored, key=lambda pair: (pair[0], _expr_key(pair[1])))
        current = _fit_score(X, y, selected) if selected else _fit_score(X, y, ())
        if score >= current - 1e-10:
            break
        selected += (best,)
        remaining.remove(best)
    return selected, _fit_score(X, y, selected), evaluations


def run_structure_discovery_benchmark(
    config: DiscoveryConfig = DiscoveryConfig(),
    strategies: tuple[Strategy, ...] = STRATEGIES,
) -> DiscoveryReport:
    """Compare five search policies on the same synthetic compositional law.

    Ground-truth target: y = x0^2 + sin(x1) + noise. Candidate scoring uses
    train data; candidate-set choice is frozen before validation/test evaluation.
    This is a bounded operator-discovery test, not a test of unconstrained math.
    """
    if not strategies or any(s not in STRATEGIES for s in strategies):
        raise ValueError("unsupported strategy")
    trials: list[DiscoveryTrial] = []
    for seed in range(config.seeds):
        rng = np.random.default_rng(seed + 290_041)
        X = rng.uniform(-1.0, 1.0, size=(config.samples, 3))
        y = X[:, 0] ** 2 + np.sin(X[:, 1]) + rng.normal(0, config.noise_std, config.samples)
        perm = rng.permutation(config.samples)
        n_test = max(16, int(config.samples * config.test_fraction))
        n_val = max(16, int(config.samples * config.validation_fraction))
        test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test+n_val], perm[n_test+n_val:]
        Xtr, ytr = X[train_idx], y[train_idx]
        Xv, yv = X[val_idx], y[val_idx]
        Xte, yte = X[test_idx], y[test_idx]
        baseline = _fit_score(Xte, yte, ())
        for strategy in strategies:
            local_rng = np.random.default_rng(seed * 1_000_003 + STRATEGIES.index(strategy) * 1009 + 41)
            candidates = _candidate_order(strategy, local_rng, config.candidate_budget, Xtr, ytr)
            selected, _, evaluations = _greedy_select(Xtr, ytr, candidates, max_terms=2)
            validation_rmse = _fit_score(Xv, yv, selected)
            test_rmse = _fit_score(Xte, yte, selected)
            gain = (baseline - test_rmse) / baseline if baseline > 1e-12 else 0.0
            keys = {_expr_key(e) for e in selected}
            target_keys = {_expr_key(Expr("square", (variable(0),))),
                           _expr_key(Expr("sin", (variable(1),)))}
            found = len(keys & target_keys)
            trials.append(DiscoveryTrial(
                seed, strategy, test_rmse, baseline, gain, found,
                min(evaluations, config.candidate_budget * max(1, len(selected))),
                sum(e.complexity for e in selected), validation_rmse,
            ))
    summaries = []
    for strategy in strategies:
        rows = [t for t in trials if t.strategy == strategy]
        summaries.append(DiscoverySummary(
            strategy=strategy,
            mean_test_rmse=float(np.mean([r.test_rmse for r in rows])),
            mean_relative_gain=float(np.mean([r.relative_gain for r in rows])),
            component_recovery_rate=float(np.mean([r.target_components_found / 2 for r in rows])),
            mean_candidates_evaluated=float(np.mean([r.candidates_evaluated for r in rows])),
            mean_selected_complexity=float(np.mean([r.selected_complexity for r in rows])),
            trials=len(rows),
        ))
    return DiscoveryReport(
        config, tuple(trials), tuple(summaries),
        (
            "All strategies share the same expression grammar and candidate-evaluation cap.",
            "The benchmark tests search policies on one synthetic law; it cannot establish universal superiority.",
            "The grammar contains human-designed primitives, so this is not mathematics from nothing.",
            "Test RMSE is reported for evaluation only; search and greedy selection use training data.",
        ),
    )
