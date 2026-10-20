"""ECF v4.20: empirical invariance discovery for bounded expression trees.

This is a diagnostic benchmark, not a proof of mathematical invariance. A candidate
is reported invariant only relative to the supplied transformations, samples, and
numerical tolerance. Generated expressions remain interpreted by v4.19's whitelist.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np
from .v419 import Expr, evaluate

ArrayTransform = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class InvarianceReport:
    expression: Expr
    passed: bool
    max_normalized_error: float
    checks: int
    failed_transform_indices: tuple[int, ...]

def reflection(X: np.ndarray) -> np.ndarray:
    """Reflect every feature through zero: x -> -x."""
    return -np.asarray(X, dtype=float)

def translation(offset: float) -> ArrayTransform:
    """Return a deterministic translation x -> x + offset."""
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    def apply(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) + offset
    return apply

def scaling(factor: float) -> ArrayTransform:
    """Return a deterministic scaling x -> factor*x."""
    if not np.isfinite(factor) or factor == 0:
        raise ValueError("factor must be finite and non-zero")
    def apply(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) * factor
    return apply

def check_invariance(expr: Expr, X: np.ndarray,
                     transforms: Iterable[ArrayTransform], *,
                     tolerance: float = 1e-8) -> InvarianceReport:
    """Check f(T(X)) ~= f(X) for each supplied transformation T."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 1 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a non-empty finite 2-D matrix")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    reference = evaluate(expr, X)
    scale = max(float(np.sqrt(np.mean(reference ** 2))), 1.0)
    errors: list[float] = []
    failed: list[int] = []
    for i, transform in enumerate(tuple(transforms)):
        Xt = np.asarray(transform(X), dtype=float)
        if Xt.shape != X.shape or not np.all(np.isfinite(Xt)):
            raise ValueError("transform must return a finite matrix matching X")
        transformed = evaluate(expr, Xt)
        error = float(np.sqrt(np.mean((transformed - reference) ** 2)) / scale)
        errors.append(error)
        if error > tolerance:
            failed.append(i)
    maximum = max(errors, default=0.0)
    return InvarianceReport(expr, not failed, maximum, len(errors), tuple(failed))

def discover_invariants(candidates: Iterable[Expr], X: np.ndarray,
                        transforms: Iterable[ArrayTransform], *,
                        tolerance: float = 1e-8) -> tuple[InvarianceReport, ...]:
    """Return reports in candidate order; does not promote or mutate the language."""
    transforms = tuple(transforms)
    return tuple(check_invariance(expr, X, transforms, tolerance=tolerance)
                 for expr in candidates)
