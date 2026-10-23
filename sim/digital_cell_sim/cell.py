"""The digital cell's deliberately restricted observation and memory interface.

The agent receives only contact time and contacted-rod length. It never receives
rod identity, position, orientation, velocity, or the simulator's hidden state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median


@dataclass(frozen=True)
class CellObservation:
    time: float
    length: float


@dataclass(frozen=True)
class CellExperience:
    event_index: int
    observed_time: float
    observed_length: float
    predicted_time: float | None
    predicted_length: float | None
    time_absolute_error: float | None
    length_absolute_error: float | None


class DigitalCellAgent:
    """Online predictor with bounded memory and no access to hidden simulator state."""

    def __init__(self, memory_capacity: int = 1000, window: int = 5):
        if memory_capacity < 2:
            raise ValueError("memory_capacity must be >= 2")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.memory_capacity = memory_capacity
        self.window = window
        self.memory: list[CellObservation] = []
        self.experiences: list[CellExperience] = []

    def predict_next(self) -> dict[str, float | None]:
        """Predict the next contact from memory available before it occurs."""
        if not self.memory:
            return {"predicted_time": None, "predicted_length": None}
        lengths = [item.length for item in self.memory[-self.window:]]
        intervals = [
            self.memory[i].time - self.memory[i - 1].time
            for i in range(max(1, len(self.memory) - self.window), len(self.memory))
        ]
        predicted_interval = float(median(intervals)) if intervals else None
        return {
            "predicted_time": self.memory[-1].time + predicted_interval if predicted_interval is not None else None,
            "predicted_length": float(median(lengths)),
        }

    def observe(self, time: float, length: float) -> CellExperience:
        """Score a pre-observation forecast, then update memory with this observation."""
        if not math.isfinite(time) or not math.isfinite(length) or time < 0 or length <= 0:
            raise ValueError("time must be finite and >= 0; length must be finite and > 0")
        if self.memory and time < self.memory[-1].time:
            raise ValueError("observations must be nondecreasing in time")
        forecast = self.predict_next()
        predicted_time = forecast["predicted_time"]
        predicted_length = forecast["predicted_length"]
        experience = CellExperience(
            event_index=len(self.experiences),
            observed_time=float(time),
            observed_length=float(length),
            predicted_time=predicted_time,
            predicted_length=predicted_length,
            time_absolute_error=abs(time - predicted_time) if predicted_time is not None else None,
            length_absolute_error=abs(length - predicted_length) if predicted_length is not None else None,
        )
        self.experiences.append(experience)
        self.memory.append(CellObservation(float(time), float(length)))
        if len(self.memory) > self.memory_capacity:
            self.memory = self.memory[-self.memory_capacity:]
        return experience

    def metrics(self) -> dict:
        time_errors = [x.time_absolute_error for x in self.experiences if x.time_absolute_error is not None]
        length_errors = [x.length_absolute_error for x in self.experiences if x.length_absolute_error is not None]
        return {
            "observation_count": len(self.experiences),
            "time_forecast_count": len(time_errors),
            "length_forecast_count": len(length_errors),
            "time_mae": sum(time_errors) / len(time_errors) if time_errors else None,
            "length_mae": sum(length_errors) / len(length_errors) if length_errors else None,
        }

    def export(self) -> dict:
        return {
            "observation_contract": ["time", "length"],
            "memory_capacity": self.memory_capacity,
            "window": self.window,
            "memory": [asdict(item) for item in self.memory],
            "experiences": [asdict(item) for item in self.experiences],
            "metrics": self.metrics(),
            "leakage_guard": (
                "The cell agent stores only time and length. Rod IDs, positions, angles, "
                "velocities and other simulator state are excluded."
            ),
        }
