#!/usr/bin/env python3
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from flf_ecf.v418 import benchmark
p=argparse.ArgumentParser();p.add_argument("--main-runs",type=int,default=5);p.add_argument("--ablation-runs",type=int,default=3);a=p.parse_args()
s,r=benchmark(a.main_runs,a.ablation_runs)
(ROOT/"results").mkdir(exist_ok=True)
(ROOT/"results/v4.18-raw.json").write_text(json.dumps(r,indent=2));(ROOT/"results/v4.18-summary.json").write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
