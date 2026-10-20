"""ECF v4.19: bounded operator genesis from persistent prediction residuals.

Generated candidates are immutable expression trees interpreted by a whitelist
evaluator; this module never executes generated Python or shell code.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    index: int | None = None
    @property
    def complexity(self) -> int:
        return 1 + sum(arg.complexity for arg in self.args)

def variable(index: int) -> Expr:
    if index < 0:
        raise ValueError("feature index must be non-negative")
    return Expr("var", index=index)

def evaluate(expr: Expr, X: np.ndarray) -> np.ndarray:
    """Evaluate a whitelisted expression over a 2-D feature matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D feature matrix")
    if expr.op == "var":
        if expr.index is None or expr.index >= X.shape[1]:
            raise ValueError("variable index outside feature matrix")
        return X[:, expr.index]
    values = [evaluate(arg, X) for arg in expr.args]
    if expr.op == "add" and len(values) == 2:
        out = values[0] + values[1]
    elif expr.op == "mul" and len(values) == 2:
        out = values[0] * values[1]
    elif expr.op == "sin" and len(values) == 1:
        out = np.sin(values[0])
    elif expr.op == "abs" and len(values) == 1:
        out = np.abs(values[0])
    elif expr.op == "square" and len(values) == 1:
        out = values[0] ** 2
    else:
        raise ValueError(f"unsupported expression: {expr.op}/{len(values)}")
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("operator produced non-finite values")
    return out

def initial_language(n_features: int) -> tuple[Expr, ...]:
    """Initial language contains only raw feature access."""
    if n_features < 1:
        raise ValueError("at least one feature is required")
    return tuple(variable(i) for i in range(n_features))

def propose_operators(language: Iterable[Expr]) -> tuple[Expr, ...]:
    """Propose novel compositions; candidates are deterministic and deduplicated."""
    base = tuple(language)
    proposals: list[Expr] = []
    for a, b in combinations(base, 2):
        proposals.extend((Expr("add", (a, b)), Expr("mul", (a, b))))
    for a in base:
        proposals.extend((Expr("sin", (a,)), Expr("abs", (a,)), Expr("square", (a,))))
    known = set(base)
    unique = []
    for candidate in proposals:
        if candidate not in known and candidate not in unique:
            unique.append(candidate)
    return tuple(unique)

def _design(X: np.ndarray, operators: Iterable[Expr]) -> np.ndarray:
    cols = [np.ones(X.shape[0])]
    cols.extend(evaluate(op, X) for op in operators)
    return np.column_stack(cols)

def _fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                 X_eval: np.ndarray, operators: tuple[Expr, ...]) -> np.ndarray:
    A = _design(X_train, operators)
    B = _design(X_eval, operators)
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return B @ coef

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

@dataclass(frozen=True)
class PromotionReport:
    accepted: bool
    baseline_rmse: float
    candidate_rmse: float
    relative_gain: float
    operator: Expr | None
    reason: str

def validate_candidate(X_train: np.ndarray, y_train: np.ndarray,
                       X_valid: np.ndarray, y_valid: np.ndarray,
                       language: tuple[Expr, ...], candidate: Expr, *,
                       min_relative_gain: float = 0.02,
                       max_complexity: int = 8) -> PromotionReport:
    """Promote only if a candidate improves held-out prediction by a fixed margin."""
    if not 0 <= min_relative_gain < 1:
        raise ValueError("min_relative_gain must be in [0, 1)")
    if candidate in language:
        raise ValueError("candidate is already in the language")
    if candidate.complexity > max_complexity:
        return PromotionReport(False, float("nan"), float("nan"), 0.0, None, "complexity_limit")
    X_train, X_valid = np.asarray(X_train, dtype=float), np.asarray(X_valid, dtype=float)
    y_train, y_valid = np.asarray(y_train, dtype=float), np.asarray(y_valid, dtype=float)
    if X_train.ndim != 2 or X_valid.ndim != 2:
        raise ValueError("features must be 2-D")
    if X_train.shape[1] != X_valid.shape[1] or X_train.shape[1] < 1:
        raise ValueError("train/validation feature dimensions must match")
    if len(X_train) != len(y_train) or len(X_valid) != len(y_valid):
        raise ValueError("feature and target lengths must match")
    if len(X_train) < 4 or len(X_valid) < 4:
        raise ValueError("need at least four training and validation rows")
    if not all(np.all(np.isfinite(a)) for a in (X_train, X_valid, y_train, y_valid)):
        raise ValueError("inputs must be finite")
    baseline_pred = _fit_predict(X_train, y_train, X_valid, language)
    candidate_pred = _fit_predict(X_train, y_train, X_valid, language + (candidate,))
    base, score = rmse(y_valid, baseline_pred), rmse(y_valid, candidate_pred)
    gain = (base - score) / max(base, 1e-15)
    accepted = gain >= min_relative_gain
    return PromotionReport(accepted, base, score, gain,
                           candidate if accepted else None,
                           "heldout_gain" if accepted else "insufficient_heldout_gain")

def promote(language: tuple[Expr, ...], report: PromotionReport) -> tuple[Expr, ...]:
    """Pure state transition: only a validated report can change the language."""
    if not report.accepted or report.operator is None or report.operator in language:
        return language
    return language + (report.operator,)
