"""ECF v4.21: bounded, resumable research loop.

The loop orchestrates user-supplied research steps; it does not guarantee a solution.
Checkpoint writes are atomic. A step must return JSON-serializable evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import time
from typing import Any, Callable


class ResearchStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    STALLED = "stalled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STEP_LIMIT = "step_limit"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ResearchState:
    problem_id: str
    step_index: int = 0
    status: str = ResearchStatus.RUNNING.value
    best_score: float | None = None
    steps_without_progress: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    strategy_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of one research iteration.

    score is a comparable objective value where lower is better by default.
    Set success=True only when the caller's independent acceptance gate passes.
    """
    score: float | None
    success: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    strategy_changed: bool = False
    note: str = ""


@dataclass(frozen=True)
class LoopConfig:
    max_steps: int = 100
    max_seconds: float = 300.0
    max_steps_without_progress: int = 10
    min_improvement: float = 1e-9
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_steps_without_progress < 1:
            raise ValueError("step limits must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.max_seconds <= 0 or self.min_improvement < 0:
            raise ValueError("time budget must be positive and improvement non-negative")


StepFunction = Callable[[ResearchState], StepResult]


def save_checkpoint(state: ResearchState, path: str | Path) -> None:
    """Atomically persist a JSON checkpoint; replaces only the target checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = json.dumps(asdict(state), sort_keys=True, allow_nan=False)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_checkpoint(path: str | Path, *, problem_id: str | None = None) -> ResearchState:
    """Load a checkpoint and optionally ensure it belongs to the requested problem."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = ResearchState(**payload)
    if problem_id is not None and state.problem_id != problem_id:
        raise ValueError("checkpoint problem_id does not match")
    if state.status not in {item.value for item in ResearchStatus}:
        raise ValueError("checkpoint contains an unknown status")
    return state


def run_research(
    state: ResearchState,
    step_fn: StepFunction,
    *,
    config: LoopConfig = LoopConfig(),
    checkpoint_path: str | Path | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> ResearchState:
    """Run bounded iterations, preserving history and allowing safe resumption.

    A repeated non-improving score increments a stall counter. Strategy changes
    reset that counter but do not erase evidence. A successful result is accepted
    only when step_fn explicitly sets success=True.
    """
    if state.status == ResearchStatus.SUCCESS.value:
        return state
    if state.status not in {ResearchStatus.RUNNING.value, ResearchStatus.PAUSED.value,
                            ResearchStatus.STALLED.value, ResearchStatus.BUDGET_EXHAUSTED.value,
                            ResearchStatus.STEP_LIMIT.value}:
        raise ValueError("state is not resumable")
    state.status = ResearchStatus.RUNNING.value
    start = time.monotonic()
    steps_this_run = 0

    while steps_this_run < config.max_steps:
        if should_pause is not None and should_pause():
            state.status = ResearchStatus.PAUSED.value
            break
        if time.monotonic() - start >= config.max_seconds:
            state.status = ResearchStatus.BUDGET_EXHAUSTED.value
            break
        try:
            result = step_fn(state)
            if not isinstance(result, StepResult):
                raise TypeError("step_fn must return StepResult")
            if result.score is not None:
                import math
                if not math.isfinite(result.score):
                    raise ValueError("score must be finite or None")
            # Validate evidence before mutating persistent state.
            json.dumps(result.evidence, allow_nan=False)
        except Exception as exc:
            state.history.append({"step": state.step_index, "event": "step_error",
                                  "error_type": type(exc).__name__, "message": str(exc)})
            state.status = ResearchStatus.ERROR.value
            break

        prior_best = state.best_score
        improved = (result.score is not None and
                    (prior_best is None or prior_best - result.score > config.min_improvement))
        if improved:
            state.best_score = result.score
            state.steps_without_progress = 0
        elif result.strategy_changed:
            state.strategy_index += 1
            state.steps_without_progress = 0
        else:
            state.steps_without_progress += 1

        state.history.append({
            "step": state.step_index,
            "score": result.score,
            "best_score": state.best_score,
            "improved": improved,
            "strategy_index": state.strategy_index,
            "strategy_changed": result.strategy_changed,
            "success_claimed_by_acceptance_gate": result.success,
            "note": result.note,
            "evidence": result.evidence,
        })
        state.step_index += 1
        steps_this_run += 1

        if checkpoint_path is not None and state.step_index % config.checkpoint_every == 0:
            save_checkpoint(state, checkpoint_path)
        if result.success:
            state.status = ResearchStatus.SUCCESS.value
            break
        if state.steps_without_progress >= config.max_steps_without_progress:
            state.status = ResearchStatus.STALLED.value
            break

    if state.status == ResearchStatus.RUNNING.value:
        state.status = ResearchStatus.STEP_LIMIT.value
    if checkpoint_path is not None:
        save_checkpoint(state, checkpoint_path)
    return state
