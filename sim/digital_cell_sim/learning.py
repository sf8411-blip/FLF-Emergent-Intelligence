"""Leakage-resistant event-sequence baselines for the digital cell."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Iterable

from .physics import CollisionEvent


@dataclass
class ForecastScore:
    event_index: int
    actual_time: float
    predicted_time: float | None
    actual_length: float
    predicted_length: float | None
    time_absolute_error: float | None
    length_absolute_error: float | None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def evaluate_online_baselines(
    events: Iterable[CollisionEvent], window: int = 5
) -> dict:
    """Evaluate one-step-ahead forecasts without allowing the current event to leak.

    Forecasts for event k use only events with indices < k. Time prediction uses
    the median of prior inter-event intervals; length prediction uses the median
    of prior observed lengths. The first event establishes the interval origin.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    ordered = sorted(events, key=lambda e: e.time)
    if any(not math.isfinite(e.time) or not math.isfinite(e.observed_length) for e in ordered):
        raise ValueError("events must contain finite times and lengths")
    if any(ordered[i].time < ordered[i-1].time for i in range(1, len(ordered))):
        raise ValueError("events must be time ordered")
    scores: list[ForecastScore] = []
    intervals: list[float] = []
    lengths: list[float] = []
    previous_time: float | None = None
    for idx, event in enumerate(ordered):
        predicted_interval = _median(intervals[-window:])
        predicted_length = _median(lengths[-window:])
        predicted_time = previous_time + predicted_interval if previous_time is not None and predicted_interval is not None else None
        scores.append(ForecastScore(
            event_index=idx,
            actual_time=event.time,
            predicted_time=predicted_time,
            actual_length=event.observed_length,
            predicted_length=predicted_length,
            time_absolute_error=abs(event.time-predicted_time) if predicted_time is not None else None,
            length_absolute_error=abs(event.observed_length-predicted_length) if predicted_length is not None else None,
        ))
        if previous_time is not None:
            intervals.append(event.time - previous_time)
        lengths.append(event.observed_length)
        previous_time = event.time
    time_errors = [s.time_absolute_error for s in scores if s.time_absolute_error is not None]
    length_errors = [s.length_absolute_error for s in scores if s.length_absolute_error is not None]
    return {
        "event_count": len(ordered),
        "window": window,
        "forecast_count_time": len(time_errors),
        "forecast_count_length": len(length_errors),
        "time_mae": sum(time_errors)/len(time_errors) if time_errors else None,
        "length_mae": sum(length_errors)/len(length_errors) if length_errors else None,
        "scores": [asdict(s) for s in scores],
        "interpretation": (
            "Exploratory baseline only. Metrics are not evidence of learning unless "
            "compared against matched predictors on held-out initial conditions."
        ),
    }
