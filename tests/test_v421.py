import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flf_ecf.v421 import (
    LoopConfig, ResearchState, ResearchStatus, StepResult,
    load_checkpoint, run_research, save_checkpoint,
)


class TestV421(unittest.TestCase):
    def test_success_requires_explicit_acceptance_gate(self):
        state = ResearchState("toy")
        results = iter([
            StepResult(score=2.0, success=False, evidence={"trial": 1}),
            StepResult(score=1.0, success=True, evidence={"validated": True}),
        ])
        out = run_research(state, lambda s: next(results),
                           config=LoopConfig(max_steps=10))
        self.assertEqual(out.status, ResearchStatus.SUCCESS.value)
        self.assertEqual(out.step_index, 2)

    def test_stalls_after_configured_non_improving_steps(self):
        state = ResearchState("flat")
        out = run_research(
            state, lambda s: StepResult(score=1.0),
            config=LoopConfig(max_steps=50, max_steps_without_progress=3),
        )
        self.assertEqual(out.status, ResearchStatus.STALLED.value)
        self.assertEqual(out.steps_without_progress, 3)

    def test_strategy_change_resets_stall_counter(self):
        state = ResearchState("switch")
        results = iter([
            StepResult(score=1.0),
            StepResult(score=1.0, strategy_changed=True),
            StepResult(score=0.5, success=True),
        ])
        out = run_research(state, lambda s: next(results),
                           config=LoopConfig(max_steps=5, max_steps_without_progress=2))
        self.assertEqual(out.status, ResearchStatus.SUCCESS.value)
        self.assertEqual(out.strategy_index, 1)

    def test_checkpoint_round_trip_and_problem_guard(self):
        state = ResearchState("p1", step_index=4, best_score=0.25,
                              history=[{"evidence": {"ok": True}}])
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "state.json"
            save_checkpoint(state, path)
            loaded = load_checkpoint(path, problem_id="p1")
            self.assertEqual(loaded.problem_id, "p1")
            self.assertEqual(loaded.step_index, 4)
            self.assertEqual(loaded.best_score, 0.25)
            with self.assertRaises(ValueError):
                load_checkpoint(path, problem_id="p2")

    def test_resume_from_stalled_checkpoint(self):
        state = ResearchState("resume", status=ResearchStatus.STALLED.value,
                              step_index=2, best_score=4.0, steps_without_progress=2)
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "state.json"
            save_checkpoint(state, path)
            resumed = load_checkpoint(path, problem_id="resume")
            out = run_research(resumed, lambda s: StepResult(score=1.0, success=True),
                               config=LoopConfig(max_steps=2), checkpoint_path=path)
            self.assertEqual(out.status, ResearchStatus.SUCCESS.value)
            self.assertEqual(load_checkpoint(path).status, ResearchStatus.SUCCESS.value)

    def test_non_finite_score_stops_with_error(self):
        state = ResearchState("bad")
        out = run_research(state, lambda s: StepResult(score=float("nan")))
        self.assertEqual(out.status, ResearchStatus.ERROR.value)
        self.assertEqual(out.history[-1]["event"], "step_error")

    def test_step_limit_is_explicit(self):
        state = ResearchState("limit")
        out = run_research(state, lambda s: StepResult(score=None),
                           config=LoopConfig(max_steps=2, max_steps_without_progress=9))
        self.assertEqual(out.status, ResearchStatus.STEP_LIMIT.value)
        self.assertEqual(out.step_index, 2)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            LoopConfig(max_steps=0)


if __name__ == "__main__":
    unittest.main()
