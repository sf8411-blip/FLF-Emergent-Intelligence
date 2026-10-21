#!/usr/bin/env python3
"""Run v4.28 receiver-revalidated exchange scenarios and write JSON results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flf_ecf.v428 import ExchangeConfig, run_typed_exchange_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/v4.28-typed-exchange.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.first_seed < 0:
        parser.error("--seeds must be positive and --first-seed non-negative")
    report = run_typed_exchange_experiment(
        tuple(range(args.first_seed, args.first_seed + args.seeds)),
        config=ExchangeConfig(),
    )
    payload = {
        "experiment": "ECF v4.28 typed receiver-side revalidation",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "report": asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Wrote {args.output}")
    for row in report.summaries:
        print(f"{row.scenario:>22} | {row.condition:>20} | "
              f"test_e2e={row.mean_test_accuracy:.4f} | "
              f"coverage={row.selection_coverage:.3f} | "
              f"test_if_selected={row.mean_test_accuracy_when_selected if row.mean_test_accuracy_when_selected is not None else float(0):.4f} | "
              f"validation={row.mean_validation_accuracy:.4f} | "
              f"error={row.mean_threshold_error:.4f} | "
              f"false_rejected={row.mean_rejected_false_candidates:.2f} | "
              f"compute={row.mean_compute_used:.1f}")
    print("Interpretation:", " ".join(report.interpretation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
