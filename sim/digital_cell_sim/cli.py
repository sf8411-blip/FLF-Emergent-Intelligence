from __future__ import annotations

import argparse
import json
from pathlib import Path

from .learning import evaluate_online_baselines
from .physics import DigitalCellSimulation, SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the digital-cell rotating rectangle simulation.")
    parser.add_argument("--duration", type=float, default=30.0, help="simulation duration in seconds")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"), help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="optional fixed seed; omit for fresh random initial conditions")
    parser.add_argument("--rods", type=int, default=8)
    parser.add_argument("--omega", type=float, default=0.25, help="clockwise angular speed in rad/s")
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--max-length", type=float, default=1.2)
    parser.add_argument("--dt", type=float, default=1.0/120.0)
    parser.add_argument("--max-substeps", type=int, default=256)
    args = parser.parse_args()

    config = SimulationConfig(
        width=args.width, height=args.height, rod_count=args.rods,
        clockwise_omega=args.omega, max_rod_length=args.max_length,
        fixed_dt=args.dt, max_substeps=args.max_substeps, seed=args.seed,
    )
    sim = DigitalCellSimulation(config)
    def progress(time, count):
        print(f"t={time:8.3f}s  cell_contacts={count}", flush=True)

    sim.run(args.duration, progress_callback=progress)
    sim.write_outputs(args.output)
    metrics = evaluate_online_baselines(sim.events)
    (args.output / "prediction_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    summary = {
        "simulated_seconds": sim.time,
        "rod_count": len(sim.rods),
        "cell_contact_count": len(sim.events),
        "clockwise_omega_rad_s": args.omega,
        "requested_seed": args.seed,
        "effective_seed": sim.effective_seed,
        "outputs": str(args.output.resolve()),
        "prediction_metrics": {
            "offline_baseline": {
                "time_mae": metrics["time_mae"],
                "length_mae": metrics["length_mae"],
                "forecast_count_time": metrics["forecast_count_time"],
                "forecast_count_length": metrics["forecast_count_length"],
            },
            "online_cell_agent": sim.agent.metrics(),
        },
        "validation_note": (
            "Numerical regression tests are separate from this run. Adaptive substepping "
            "reduces tunnelling risk but does not prove exact continuous collision detection."
        ),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
