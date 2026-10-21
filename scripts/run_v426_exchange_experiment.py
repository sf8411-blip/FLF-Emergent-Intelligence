#!/usr/bin/env python3
"""Run the v4.26 controlled experience-exchange benchmark and write JSON results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path

import numpy as np

from flf_ecf.v426 import ExperimentConfig, run_exchange_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50,
                        help="number of consecutive independent seeds (default: 50)")
    parser.add_argument("--first-seed", type=int, default=0,
                        help="first seed; subsequent seeds are consecutive")
    parser.add_argument("--output", type=Path,
                        default=Path("results/v4.26-exchange-benchmark.json"),
                        help="JSON output path")
    args = parser.parse_args()
    if args.seeds < 1 or args.first_seed < 0:
        parser.error("--seeds must be positive and --first-seed non-negative")

    seeds = tuple(range(args.first_seed, args.first_seed + args.seeds))
    report = run_exchange_experiment(seeds=seeds, config=ExperimentConfig())
    payload = {
        "experiment": "ECF v4.26 controlled experience exchange",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "report": asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    for summary in report.summaries:
        print(
            f"{summary.condition:>20}: "
            f"test_accuracy={summary.mean_test_accuracy:.4f}, "
            f"threshold_error={summary.mean_threshold_error:.4f}, "
            f"candidates={summary.mean_candidates_evaluated:.2f}"
        )
    for condition, gain in report.paired_accuracy_gain_vs_isolated:
        print(f"paired accuracy gain vs isolated ({condition}): {gain:+.4f}")
    print("Interpretation:", " ".join(report.interpretation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
