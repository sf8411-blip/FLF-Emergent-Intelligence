"""Rigid-body simulation of a rotating rectangular arena and a digital cell.

Uses Pymunk/Chipmunk2D for rigid-body contact resolution. Adaptive substepping
limits maximum linear/angular feature displacement; it is not a mathematical
proof of zero tunnelling for arbitrary velocities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import pymunk


ROD = 1
CELL = 2
WALL = 3


@dataclass(frozen=True)
class SimulationConfig:
    width: float = 12.0
    height: float = 10.0
    cell_radius: float = 0.55
    rod_radius: float = 0.025
    wall_radius: float = 0.035
    rod_count: int = 8
    min_rod_length: float = 0.35
    max_rod_length: float = 1.2
    min_speed: float = 0.35
    max_speed: float = 1.8
    max_angular_speed: float = 1.5
    clockwise_omega: float = 0.25
    elasticity: float = 1.0
    friction: float = 0.0
    fixed_dt: float = 1.0 / 120.0
    max_substeps: int = 256
    seed: int = 17
    max_placement_attempts: int = 5000
    enforce_total_length_constraint: bool = True

    def validate(self) -> None:
        positive = {
            "width": self.width, "height": self.height,
            "cell_radius": self.cell_radius, "rod_radius": self.rod_radius,
            "wall_radius": self.wall_radius, "min_rod_length": self.min_rod_length,
            "max_rod_length": self.max_rod_length, "min_speed": self.min_speed,
            "max_speed": self.max_speed, "fixed_dt": self.fixed_dt,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.rod_count < 1:
            raise ValueError("rod_count must be >= 1")
        if self.min_rod_length > self.max_rod_length:
            raise ValueError("min_rod_length must be <= max_rod_length")
        if self.max_rod_length > min(self.width, self.height) / 6.0:
            raise ValueError("max_rod_length must be <= min(width, height)/6 (one-third of center-to-nearest-wall distance)")
        if self.cell_radius + self.rod_radius >= min(self.width, self.height) / 2:
            raise ValueError("cell does not fit inside arena")
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must be <= max_speed")
        if not 0 <= self.elasticity <= 1.5:
            raise ValueError("elasticity must be between 0 and 1.5")
        if self.friction < 0 or not math.isfinite(self.friction):
            raise ValueError("friction must be finite and >= 0")
        if self.max_substeps < 1:
            raise ValueError("max_substeps must be >= 1")
        if self.max_placement_attempts < self.rod_count:
            raise ValueError("max_placement_attempts is too small")
        if self.enforce_total_length_constraint:
            target = (min(self.width, self.height) + max(self.width, self.height)) / 4.0 - self.cell_radius
            if not self.rod_count * self.min_rod_length <= target <= self.rod_count * self.max_rod_length:
                raise ValueError(
                    "rod_count/min_rod_length/max_rod_length cannot satisfy the total-length constraint; "
                    f"target total length is {target:.6g}"
                )


@dataclass
class RodRecord:
    rod_id: int
    length: float
    mass: float
    radius: float


@dataclass
class CollisionEvent:
    time: float
    event: str
    rod_id: int
    observed_length: float
    other: str = "cell"


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-20:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def _segment_endpoints(x: float, y: float, angle: float, length: float) -> tuple[tuple[float, float], tuple[float, float]]:
    dx, dy = math.cos(angle) * length / 2.0, math.sin(angle) * length / 2.0
    return (x - dx, y - dy), (x + dx, y + dy)


def _segments_distance(a0, a1, b0, b1) -> float:
    # In 2-D, non-parallel segments intersect iff endpoint-to-segment minimum is zero.
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    o1, o2 = orient(a0, a1, b0), orient(a0, a1, b1)
    o3, o4 = orient(b0, b1, a0), orient(b0, b1, a1)
    if o1 * o2 <= 0 and o3 * o4 <= 0:
        # Bounding-box check avoids treating disjoint collinear segments as intersecting.
        if (max(min(a0[0], a1[0]), min(b0[0], b1[0])) <= min(max(a0[0], a1[0]), max(b0[0], b1[0])) + 1e-12
            and max(min(a0[1], a1[1]), min(b0[1], b1[1])) <= min(max(a0[1], a1[1]), max(b0[1], b1[1])) + 1e-12):
            return 0.0
    return min(
        _point_segment_distance(a0, b0, b1), _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1), _point_segment_distance(b1, a0, a1),
    )


class DigitalCellSimulation:
    """Deterministic seeded arena simulation with observable cell contacts."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.config.validate()
        self.rng = random.Random(self.config.seed)
        self.space = pymunk.Space(threaded=False)
        self.space.gravity = (0.0, 0.0)
        self.space.iterations = 30
        self.space.collision_slop = 0.001
        self.space.collision_bias = (1.0 - 0.1) ** 60
        self.time = 0.0
        self.events: list[CollisionEvent] = []
        self._last_cell_event_time: dict[int, float] = {}
        self.rods: list[RodRecord] = []
        self._meta_by_body: dict[Any, RodRecord] = {}
        self._build_cell()
        self._build_walls()
        self._build_rods()
        self.space.on_collision(ROD, CELL, begin=self._on_cell_collision)

    def _build_cell(self) -> None:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (0.0, 0.0)
        shape = pymunk.Circle(body, self.config.cell_radius)
        shape.collision_type = CELL
        shape.elasticity = self.config.elasticity
        shape.friction = self.config.friction
        shape.filter = pymunk.ShapeFilter(categories=0b10, mask=0b01)
        self.space.add(body, shape)
        self.cell_body = body
        self.cell_shape = shape

    def _build_walls(self) -> None:
        c = self.config
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (0.0, 0.0)
        # Negative angular velocity means clockwise in standard x-right/y-up coordinates.
        body.angular_velocity = -c.clockwise_omega
        hw, hh = c.width / 2.0, c.height / 2.0
        vertices = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        shapes = []
        for i in range(4):
            shape = pymunk.Segment(body, vertices[i], vertices[(i + 1) % 4], c.wall_radius)
            shape.collision_type = WALL
            shape.elasticity = c.elasticity
            shape.friction = c.friction
            shape.filter = pymunk.ShapeFilter(categories=0b100, mask=0b001)
            shapes.append(shape)
        self.space.add(body, *shapes)
        self.wall_body = body
        self.wall_shapes = shapes

    def _sample_lengths(self) -> list[float]:
        c = self.config
        lengths = [self.rng.uniform(c.min_rod_length, c.max_rod_length) for _ in range(c.rod_count)]
        if not c.enforce_total_length_constraint:
            return lengths
        target = (min(c.width, c.height) + max(c.width, c.height)) / 4.0 - c.cell_radius
        # Project random lengths onto the bounded sum constraint by iteratively
        # sharing the remaining difference among lengths with available capacity.
        for _ in range(1000):
            difference = target - sum(lengths)
            if abs(difference) < 1e-10:
                break
            if difference > 0:
                eligible = [i for i, value in enumerate(lengths) if value < c.max_rod_length - 1e-12]
                if not eligible:
                    break
                share = difference / len(eligible)
                for i in eligible:
                    lengths[i] += min(share, c.max_rod_length - lengths[i])
            else:
                eligible = [i for i, value in enumerate(lengths) if value > c.min_rod_length + 1e-12]
                if not eligible:
                    break
                share = (-difference) / len(eligible)
                for i in eligible:
                    lengths[i] -= min(share, lengths[i] - c.min_rod_length)
        if abs(sum(lengths) - target) > 1e-7:
            raise RuntimeError("Could not satisfy the configured total rod-length constraint")
        return lengths

    def _build_rods(self) -> None:
        c = self.config
        lengths = self._sample_lengths()
        minx, maxx = -c.width / 2 + c.max_rod_length / 2 + c.rod_radius, c.width / 2 - c.max_rod_length / 2 - c.rod_radius
        miny, maxy = -c.height / 2 + c.max_rod_length / 2 + c.rod_radius, c.height / 2 - c.max_rod_length / 2 - c.rod_radius
        placed: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for rod_id in range(c.rod_count):
            success = False
            length = lengths[rod_id]
            for _ in range(c.max_placement_attempts):
                angle = self.rng.uniform(-math.pi, math.pi)
                x, y = self.rng.uniform(minx, maxx), self.rng.uniform(miny, maxy)
                a, b = _segment_endpoints(x, y, angle, length)
                # Keep the entire capsule inside the unrotated initial rectangle.
                if min(a[0], b[0]) < -c.width/2 + c.rod_radius or max(a[0], b[0]) > c.width/2 - c.rod_radius:
                    continue
                if min(a[1], b[1]) < -c.height/2 + c.rod_radius or max(a[1], b[1]) > c.height/2 - c.rod_radius:
                    continue
                if math.hypot(x, y) < c.cell_radius + c.rod_radius + length/2 + 0.08:
                    continue
                if any(_segments_distance(a, b, p, q) < c.rod_radius + rr + 0.03 for p, q, rr in placed):
                    continue
                mass = max(length, 0.05)
                moment = pymunk.moment_for_segment(mass, (-length/2, 0), (length/2, 0), c.rod_radius)
                body = pymunk.Body(mass, moment)
                body.position = (x, y)
                body.angle = angle
                speed = self.rng.uniform(c.min_speed, c.max_speed)
                velocity_angle = self.rng.uniform(-math.pi, math.pi)
                body.velocity = (speed * math.cos(velocity_angle), speed * math.sin(velocity_angle))
                body.angular_velocity = self.rng.uniform(-c.max_angular_speed, c.max_angular_speed)
                # Chipmunk does not reliably provide segment-vs-segment contacts for
                # all capsule configurations. Represent each finite-thickness rod as a
                # rectangle plus two end caps so rod-rod contacts use supported shape pairs.
                vertices = [
                    (-length/2, -c.rod_radius), (length/2, -c.rod_radius),
                    (length/2, c.rod_radius), (-length/2, c.rod_radius),
                ]
                shapes = [
                    pymunk.Poly(body, vertices),
                    pymunk.Circle(body, c.rod_radius, offset=(-length/2, 0)),
                    pymunk.Circle(body, c.rod_radius, offset=(length/2, 0)),
                ]
                for shape in shapes:
                    shape.collision_type = ROD
                    shape.elasticity = c.elasticity
                    shape.friction = c.friction
                    shape.filter = pymunk.ShapeFilter(categories=0b001, mask=0b111)
                record = RodRecord(rod_id=rod_id, length=length, mass=mass, radius=c.rod_radius)
                self.space.add(body, *shapes)
                self.rods.append(record)
                self._meta_by_body[body] = record
                placed.append((a, b, c.rod_radius))
                success = True
                break
            if not success:
                raise RuntimeError(
                    f"Could not place rod {rod_id}; reduce rod_count/lengths or increase arena size"
                )

    def _on_cell_collision(self, arbiter, space, data):
        rod = None
        for shape in arbiter.shapes:
            if shape.collision_type == ROD:
                rod = self._meta_by_body.get(shape.body)
                break
        if rod is not None:
            last_time = self._last_cell_event_time.get(rod.rod_id, -math.inf)
            # A capsule consists of multiple shapes. Deduplicate simultaneous
            # shape-pair callbacks into one observable contact event per rod.
            if self.time - last_time > max(self.config.fixed_dt * 0.5, 1e-8):
                self.events.append(CollisionEvent(
                    time=float(self.time), event="cell_collision",
                    rod_id=rod.rod_id, observed_length=float(rod.length), other="cell",
                ))
                self._last_cell_event_time[rod.rod_id] = float(self.time)
        return True

    def _max_contact_speed(self) -> float:
        c = self.config
        maximum = abs(self.wall_body.angular_velocity) * math.hypot(c.width/2, c.height/2)
        for body, record in self._meta_by_body.items():
            maximum = max(maximum, body.velocity.length + abs(body.angular_velocity) * (record.length / 2 + record.radius))
        return maximum

    def step(self, duration: float | None = None) -> None:
        dt = self.config.fixed_dt if duration is None else duration
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("duration must be finite and > 0")
        # Adaptive substeps bound translation of the fastest point to half the smallest
        # collision feature. If the required resolution exceeds the configured limit,
        # fail loudly rather than silently running a known under-resolved step.
        feature = min(self.config.rod_radius, self.config.wall_radius, self.config.cell_radius)
        max_move = feature * 0.5
        required = max(1, math.ceil(self._max_contact_speed() * dt / max_move))
        if required > self.config.max_substeps:
            raise RuntimeError(
                f"step needs {required} substeps, exceeding max_substeps={self.config.max_substeps}; "
                "reduce dt/speeds or raise max_substeps"
            )
        h = dt / required
        for _ in range(required):
            self.time += h
            self.space.step(h)
            self._validate_finite_state()

    def _validate_finite_state(self) -> None:
        bodies = [self.wall_body, *self._meta_by_body.keys()]
        for body in bodies:
            values = (body.position.x, body.position.y, body.velocity.x, body.velocity.y, body.angle, body.angular_velocity)
            if not all(math.isfinite(float(v)) for v in values):
                raise FloatingPointError("Non-finite rigid-body state detected")

    def run(self, duration: float, *, progress_callback=None) -> list[CollisionEvent]:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and > 0")
        end = self.time + duration
        steps = 0
        while self.time + 1e-12 < end:
            self.step(min(self.config.fixed_dt, end - self.time))
            steps += 1
            if progress_callback and steps % 100 == 0:
                progress_callback(self.time, len(self.events))
        return list(self.events)

    def snapshot(self) -> dict[str, Any]:
        c = self.config
        rods = []
        for body, record in self._meta_by_body.items():
            rods.append({
                **asdict(record),
                "x": float(body.position.x), "y": float(body.position.y),
                "angle": float(body.angle), "vx": float(body.velocity.x),
                "vy": float(body.velocity.y), "angular_velocity": float(body.angular_velocity),
            })
        return {
            "time": self.time,
            "config": asdict(c),
            "wall_angle": float(self.wall_body.angle),
            "cell": {"x": 0.0, "y": 0.0, "radius": c.cell_radius},
            "rods": rods,
            "event_count": len(self.events),
        }

    def write_outputs(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        (out / "snapshot.json").write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        (out / "collision_events.json").write_text(
            json.dumps([asdict(e) for e in self.events], indent=2), encoding="utf-8"
        )
        import csv
        with (out / "collision_events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "event", "rod_id", "observed_length", "other"])
            writer.writeheader()
            writer.writerows(asdict(e) for e in self.events)
