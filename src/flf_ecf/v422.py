"""ECF v4.22: deterministic counterexample search for empirical invariance claims.

Random search can refute a proposed invariance by finding a witness; failure to
find one is not a proof. Expressions are evaluated by v4.19's whitelist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .v419 import Expr, evaluate
from .v420 import ArrayTransform


@dataclass(frozen=True)
class Counterexample:
    sample_index: int
    input_point: tuple[float, ...]
    transformed_point: tuple[float, ...]
    original_value: float
    transformed_value: float
    normalized_error: float


@dataclass(frozen=True)
class CounterexampleReport:
    expression: Expr
    searched_samples: int
    tolerance: float
    seed: int
    counterexamples: tuple[Counterexample, ...]
    max_normalized_error: float

    @property
    def refuted(self) -> bool:
        return bool(self.counterexamples)


def search_counterexamples(
    expr: Expr,
    transform: ArrayTransform,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    *,
    n_samples: int = 1000,
    tolerance: float = 1e-8,
    max_counterexamples: int = 5,
    seed: int = 0,
) -> CounterexampleReport:
    """Search a bounded box for points where f(T(x)) differs from f(x).

    lower/upper may be scalars or per-feature vectors. The search is reproducible
    for a fixed seed and NumPy version. A transform must preserve matrix shape.
    The normalized error is abs(f(T(x))-f(x))/max(abs(f(x)), 1).
    """
    if n_samples < 1 or max_counterexamples < 1:
        raise ValueError("sample and counterexample limits must be positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.ndim > 1 or hi.ndim > 1:
        raise ValueError("bounds must be scalars or one-dimensional vectors")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("bounds must be finite")
    if np.any(lo >= hi):
        raise ValueError("each lower bound must be less than its upper bound")

    # Infer feature count from the expression's highest variable index.
    def max_var(node: Expr) -> int:
        own = node.index if node.op == "var" and node.index is not None else -1
        return max([own, *(max_var(arg) for arg in node.args)])
    n_features = max_var(expr) + 1
    if n_features < 1:
        raise ValueError("expression must reference at least one feature")
    try:
        lo = np.broadcast_to(lo, (n_features,))
        hi = np.broadcast_to(hi, (n_features,))
    except ValueError as exc:
        raise ValueError("bounds must be scalar or match expression feature count") from exc

    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, size=(n_samples, n_features))
    Xt = np.asarray(transform(X), dtype=float)
    if Xt.shape != X.shape or not np.all(np.isfinite(Xt)):
        raise ValueError("transform must return a finite matrix matching input shape")
    original = evaluate(expr, X)
    changed = evaluate(expr, Xt)
    errors = np.abs(changed - original) / np.maximum(np.abs(original), 1.0)
    witnesses = []
    for i in np.flatnonzero(errors > tolerance)[:max_counterexamples]:
        witnesses.append(Counterexample(
            sample_index=int(i),
            input_point=tuple(float(v) for v in X[i]),
            transformed_point=tuple(float(v) for v in Xt[i]),
            original_value=float(original[i]),
            transformed_value=float(changed[i]),
            normalized_error=float(errors[i]),
        ))
    return CounterexampleReport(
        expression=expr,
        searched_samples=n_samples,
        tolerance=float(tolerance),
        seed=int(seed),
        counterexamples=tuple(witnesses),
        max_normalized_error=float(np.max(errors, initial=0.0)),
    )
