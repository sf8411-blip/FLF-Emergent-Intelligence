"""Repeated experiments with fresh initial conditions by default.

Use --paired-seeds only when deliberately matching initial conditions across
environment treatments. Default mode gives every run a distinct random seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import random
from pathlib import Path
from statistics import mean, pstdev

from digital_cell_sim import DigitalCellSimulation, SimulationConfig, evaluate_online_baselines


def _new_seed(rng: random.Random | None, used: set[int]) -> int:
    while True:
        candidate = rng.getrandbits(128) if rng is not None else secrets.randbits(128)
        if candidate not in used:
            used.add(candidate)
            return candidate


def run_sweep(
    duration: float,
    replicates: int,
    omegas: list[float],
    rod_count: int,
    *,
    seed: int | None = None,
    paired_seeds: bool = False,
) -> dict:
    if duration <= 0 or replicates < 1 or rod_count < 1 or not omegas:
        raise ValueError("duration, replicates, rod_count and omegas must be valid")
    seed_rng = random.Random(seed) if seed is not None else None
    used_seeds: set[int] = set()
    paired = [_new_seed(seed_rng, used_seeds) for _ in range(replicates)] if paired_seeds else []
    rows = []
    for omega in omegas:
        for replicate in range(replicates):
            run_seed = paired[replicate] if paired_seeds else _new_seed(seed_rng, used_seeds)
            config = SimulationConfig(clockwise_omega=omega, seed=run_seed, rod_count=rod_count)
            sim = DigitalCellSimulation(config)
            initial_state = sim.snapshot()["rods"]
            initial_hash = hashlib.sha256(
                json.dumps(initial_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            sim.run(duration)
            metrics = evaluate_online_baselines(sim.events)
            rows.append({
                "omega": omega,
                "replicate": replicate,
                "seed": run_seed,
                "initial_conditions_sha256": initial_hash,
                "paired_initial_conditions": paired_seeds,
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
        group = [row for row in rows if row["omega"] == omega]
        time_errors = [row["time_mae"] for row in group if row["time_mae"] is not None]
        length_errors = [row["length_mae"] for row in group if row["length_mae"] is not None]
        groups.append({
            "omega": omega,
            "runs": len(group),
            "mean_cell_contacts": mean(row["cell_contacts"] for row in group),
            "sd_cell_contacts": pstdev(row["cell_contacts"] for row in group),
            "mean_time_mae": mean(time_errors) if time_errors else None,
            "mean_length_mae": mean(length_errors) if length_errors else None,
        })
    return {
        "duration_per_run": duration,
        "replicates_per_omega": replicates,
        "base_seed": seed,
        "seed_policy": "paired across omega" if paired_seeds else "unique per run",
        "omegas_clockwise_rad_s": omegas,
        "rod_count": rod_count,
        "runs": rows,
        "aggregates": groups,
        "interpretation": (
            "Descriptive synthetic benchmark only. Default runs use unique random initial states. "
            "Paired mode deliberately reuses each replicate's initial state across omega values "
            "to isolate the effect of rotation. Neither mode proves chaos or AGI."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--rods", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="optional base seed for a reproducible sweep")
    parser.add_argument("--paired-seeds", action="store_true", help="reuse each replicate's initial conditions across omega values")
    parser.add_argument("--output", type=Path, default=Path("runs/sweep"))
    args = parser.parse_args()
    if args.duration <= 0 or args.replicates < 1 or not args.omegas or args.rods < 1:
        parser.error("duration and rods must be positive; replicates must be >= 1 and omegas cannot be empty")
    result = run_sweep(
        args.duration, args.replicates, args.omegas, args.rods,
        seed=args.seed, paired_seeds=args.paired_seeds,
    )
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
