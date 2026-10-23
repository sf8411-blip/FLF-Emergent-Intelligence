import json
import math
import tempfile
import unittest
from pathlib import Path

from digital_cell_sim.learning import evaluate_online_baselines
from digital_cell_sim.physics import (
    CollisionEvent, DigitalCellSimulation, SimulationConfig, _segments_distance,
)


class GeometryTests(unittest.TestCase):
    def test_crossing_segments_have_zero_distance(self):
        d = _segments_distance((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_disjoint_segments_distance(self):
        d = _segments_distance((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0))
        self.assertAlmostEqual(d, math.sqrt(2.0), places=8)

    def test_length_bound_is_enforced(self):
        with self.assertRaises(ValueError):
            SimulationConfig(width=8, height=8, max_rod_length=1.5).validate()


class SimulationTests(unittest.TestCase):
    def make_sim(self):
        return DigitalCellSimulation(SimulationConfig(
            width=8, height=8, cell_radius=0.5, rod_radius=0.03, wall_radius=0.04,
            rod_count=1, min_rod_length=0.8, max_rod_length=1.0,
            min_speed=0.1, max_speed=1.0, max_angular_speed=0.2,
            clockwise_omega=0.0, fixed_dt=1/240, max_substeps=256, seed=4,
            enforce_total_length_constraint=False,
        ))

    def test_initial_geometry_has_no_penetrations(self):
        sim = self.make_sim()
        diagnostics = sim.geometry_diagnostics()
        self.assertAlmostEqual(diagnostics["max_wall_violation"], 0.0, places=10)
        self.assertAlmostEqual(diagnostics["max_cell_penetration"], 0.0, places=10)
        self.assertAlmostEqual(diagnostics["max_rod_pair_penetration"], 0.0, places=10)

    def test_simulation_advances_and_state_stays_finite(self):
        sim = self.make_sim()
        sim.run(0.25)
        self.assertGreater(sim.time, 0.24)
        snap = sim.snapshot()
        for rod in snap["rods"]:
            for key in ("x", "y", "angle", "vx", "vy", "angular_velocity"):
                self.assertTrue(math.isfinite(rod[key]), (key, rod[key]))

    def test_cell_contact_is_recorded(self):
        sim = self.make_sim()
        body = next(iter(sim._meta_by_body))
        record = sim._meta_by_body[body]
        body.position = (0.0, sim.config.cell_radius + record.length / 2 + 0.07)
        body.angle = -math.pi / 2
        body.velocity = (0.0, -1.0)
        body.angular_velocity = 0.0
        sim.run(0.6)
        self.assertTrue(sim.events, "expected a cell collision event")
        self.assertEqual(sim.events[0].event, "cell_collision")
        self.assertAlmostEqual(sim.events[0].observed_length, record.length)




    def test_finite_thickness_rod_uses_supported_contact_shapes(self):
        sim = self.make_sim()
        body = next(iter(sim._meta_by_body))
        shapes = [shape for shape in sim.space.shapes if shape.body == body]
        self.assertEqual(len(shapes), 3)
        self.assertEqual(sum(isinstance(shape, __import__("pymunk").Poly) for shape in shapes), 1)
        self.assertEqual(sum(isinstance(shape, __import__("pymunk").Circle) for shape in shapes), 2)

    def test_rod_bounces_from_fixed_wall(self):
        sim = self.make_sim()
        body = next(iter(sim._meta_by_body))
        body.position = (3.4, 0.0)
        body.angle = 0.0
        body.velocity = (1.0, 0.0)
        body.angular_velocity = 0.0
        sim.run(0.3)
        self.assertLess(body.velocity.x, 0.0, "rod should reverse horizontal velocity after wall contact")

    def test_two_rods_collide_without_passing_through(self):
        sim = DigitalCellSimulation(SimulationConfig(
            width=8, height=8, cell_radius=0.5, rod_radius=0.03, wall_radius=0.04,
            rod_count=2, min_rod_length=0.8, max_rod_length=1.0,
            min_speed=0.1, max_speed=1.0, max_angular_speed=0.2,
            clockwise_omega=0.0, fixed_dt=1/240, max_substeps=256, seed=9,
            enforce_total_length_constraint=False,
        ))
        bodies = list(sim._meta_by_body.keys())
        left, right = bodies[0], bodies[1]
        left.position, right.position = (-0.6, 1.5), (0.6, 1.5)
        left.angle = right.angle = 0.0
        left.velocity, right.velocity = (1.0, 0.0), (-1.0, 0.0)
        left.angular_velocity = right.angular_velocity = 0.0
        sim.run(0.35)
        self.assertLess(left.velocity.x, 0.0)
        self.assertGreater(right.velocity.x, 0.0)
        self.assertLess(left.position.x, right.position.x, "rod centers must not cross")

    def test_variable_rod_counts_respect_total_length_constraint(self):
        for count in (5, 8, 10, 12):
            with self.subTest(rod_count=count):
                sim = DigitalCellSimulation(SimulationConfig(seed=100 + count, rod_count=count))
                total = sum(rod.length for rod in sim.rods)
                target = (min(sim.config.width, sim.config.height) + max(sim.config.width, sim.config.height)) / 4 - sim.config.cell_radius
                self.assertEqual(len(sim.rods), count)
                self.assertAlmostEqual(total, target, places=7)

    def test_default_total_length_constraint(self):
        sim = DigitalCellSimulation(SimulationConfig(seed=21))
        total = sum(rod.length for rod in sim.rods)
        target = (min(sim.config.width, sim.config.height) + max(sim.config.width, sim.config.height)) / 4 - sim.config.cell_radius
        self.assertAlmostEqual(total, target, places=7)
        self.assertTrue(all(sim.config.min_rod_length <= rod.length <= sim.config.max_rod_length for rod in sim.rods))

    def test_unseeded_runs_have_distinct_initial_conditions(self):
        a, b = DigitalCellSimulation(), DigitalCellSimulation()
        self.assertNotEqual(a.effective_seed, b.effective_seed)
        self.assertNotEqual(a.snapshot()["rods"], b.snapshot()["rods"])

    def test_outputs_include_reproducible_initial_state_and_svg(self):
        sim = self.make_sim()
        with tempfile.TemporaryDirectory() as directory:
            sim.run(0.02)
            sim.write_outputs(directory)
            root = Path(directory)
            self.assertTrue((root / "initial_conditions.json").is_file())
            self.assertTrue((root / "environment_snapshot.svg").is_file())
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            initial = json.loads((root / "initial_conditions.json").read_text(encoding="utf-8"))
            self.assertEqual(config["effective_seed"], sim.effective_seed)
            self.assertEqual(initial["effective_seed"], sim.effective_seed)
            self.assertIn("<svg", (root / "environment_snapshot.svg").read_text(encoding="utf-8"))

    def test_seed_reproduces_initial_conditions(self):
        a, b = self.make_sim(), self.make_sim()
        sa, sb = a.snapshot(), b.snapshot()
        self.assertEqual(len(sa["rods"]), len(sb["rods"]))
        for ra, rb in zip(sa["rods"], sb["rods"]):
            for key in ("x", "y", "angle", "vx", "vy", "angular_velocity", "length"):
                self.assertAlmostEqual(ra[key], rb[key], places=12)


class LearningTests(unittest.TestCase):
    def test_predictor_uses_only_prior_events(self):
        events = [
            CollisionEvent(1.0, "cell_collision", 0, 0.5),
            CollisionEvent(3.0, "cell_collision", 1, 0.7),
            CollisionEvent(6.0, "cell_collision", 2, 0.9),
        ]
        result = evaluate_online_baselines(events, window=2)
        self.assertEqual(result["event_count"], 3)
        self.assertEqual(result["forecast_count_time"], 1)
        self.assertEqual(result["forecast_count_length"], 2)
        # Event 3's forecast uses the only previously observed interval (2 seconds).
        self.assertAlmostEqual(result["scores"][2]["predicted_time"], 5.0)
        self.assertAlmostEqual(result["scores"][2]["predicted_length"], 0.6)

    def test_empty_event_stream(self):
        result = evaluate_online_baselines([])
        self.assertEqual(result["event_count"], 0)
        self.assertIsNone(result["time_mae"])
        self.assertIsNone(result["length_mae"])


if __name__ == "__main__":
    unittest.main()
