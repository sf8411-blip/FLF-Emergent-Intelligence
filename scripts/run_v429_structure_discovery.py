#!/usr/bin/env python3
"""Run the v4.29 structure-discovery strategy benchmark and save raw trials."""
from __future__ import annotations
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flf_ecf.v429 import DiscoveryConfig, run_structure_discovery_benchmark

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--samples", type=int, default=384)
    p.add_argument("--candidate-budget", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("results/v4.29-structure-discovery.json"))
    a = p.parse_args()
    report = run_structure_discovery_benchmark(
        DiscoveryConfig(seeds=a.seeds, samples=a.samples, candidate_budget=a.candidate_budget)
    )
    payload = {"experiment": "ECF v4.29 structure-discovery strategy comparison",
               "report": asdict(report)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"Wrote {a.output}")
    for row in report.summaries:
        print(f"{row.strategy:>18} | test_rmse={row.mean_test_rmse:.5f} | "
              f"gain={row.mean_relative_gain:.3f} | component_recovery={row.component_recovery_rate:.3f} | "
              f"candidates={row.mean_candidates_evaluated:.1f} | complexity={row.mean_selected_complexity:.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
