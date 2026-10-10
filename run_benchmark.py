#!/usr/bin/env python3
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v416 import benchmark

p = argparse.ArgumentParser(description="Run the ECF v4.16 anonymous-event benchmark.")
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--raw-out", default="results/v4.16-raw.json")
p.add_argument("--summary-out", default="results/v4.16-summary.json")
a = p.parse_args()
summary, rows = benchmark(a.seeds)
raw_path = ROOT / a.raw_out
summary_path = ROOT / a.summary_out
raw_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.parent.mkdir(parents=True, exist_ok=True)
raw_path.write_text(json.dumps(rows, indent=2))
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
