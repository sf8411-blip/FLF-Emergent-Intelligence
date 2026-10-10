#!/usr/bin/env python3
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flf_ecf.v417 import benchmark

p = argparse.ArgumentParser(description="Run ECF v4.17 continuous anonymous-stream benchmark.")
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--raw-out", default="results/v4.17-raw.json")
p.add_argument("--summary-out", default="results/v4.17-summary.json")
a = p.parse_args()
summary, rows = benchmark(a.seeds)
raw = ROOT / a.raw_out
summary_path = ROOT / a.summary_out
raw.parent.mkdir(parents=True, exist_ok=True)
summary_path.parent.mkdir(parents=True, exist_ok=True)
raw.write_text(json.dumps(rows, indent=2))
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
