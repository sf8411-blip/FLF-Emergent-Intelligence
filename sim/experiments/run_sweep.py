"""Matched-seed sweep for static versus rotating arenas.

This experiment reports observed event/prediction metrics only. It does not
classify a trajectory as chaotic or claim learning from a small event stream.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from digital_cell_sim import DigitalCellSimulation, SimulationConfig, evaluate_online_baselines


def run_sweep(duration: float, seeds: list[int], omegas: list[float], rod_count: int) -> dict:
    rows = []
    for omega in omegas:
        for seed in seeds:
            config = SimulationConfig(
                clockwise_omega=omega,
                seed=seed,
                rod_count=rod_count,
            )
            sim = DigitalCellSimulation(config)
            sim.run(duration)
            metrics = evaluate_online_baselines(sim.events)
            rows.append({
                "omega": omega,
                "seed": seed,
                "duration": sim.time,
                "rod_count": len(sim.rods),
                "cell_contacts": len(sim.events),
                "time_mae": metrics["time_mae"],
                "length_mae": metrics["length_mae"],
                "time_forecasts": metrics["forecast_count_time"],
                "length_forecasts": metrics["forecast_count_length"],
            })
    groups = []
    for omega in omegas:
        group = [r for r in rows if r["omega"] == omega]
        groups.append({
            "omega": omega,
            "runs": len(group),
            "mean_cell_contacts": mean(r["cell_contacts"] for r in group),
            "sd_cell_contacts": pstdev(r["cell_contacts"] for r in group),
            "mean_time_mae": mean([r["time_mae"] for r in group if r["time_mae"] is not None]) if any(r["time_mae"] is not None for r in group) else None,
            "mean_length_mae": mean([r["length_mae"] for r in group if r["length_mae"] is not None]) if any(r["length_mae"] is not None for r in group) else None,
        })
    return {
        "duration_per_run": duration,
        "seeds": seeds,
        "omegas_clockwise_rad_s": omegas,
        "rod_count": rod_count,
        "runs": rows,
        "aggregates": groups,
        "interpretation": (
            "Descriptive synthetic benchmark only. Compare matched seeds, report uncertainty, "
            "and use held-out conditions before claiming predictive generalization. No chaos "
            "or AGI claim is implied by these metrics."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 31, 47])
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--rods", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/sweep"))
    args = parser.parse_args()
    if args.duration <= 0 or not args.seeds or not args.omegas or args.rods < 1:
        parser.error("duration and rods must be positive; seeds and omegas cannot be empty")
    result = run_sweep(args.duration, args.seeds, args.omegas, args.rods)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweep_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "sweep_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["runs"][0].keys()))
        writer.writeheader()
        writer.writerows(result["runs"])
    print(json.dumps(result["aggregates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
