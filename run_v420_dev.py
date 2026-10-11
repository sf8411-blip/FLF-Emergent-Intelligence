#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v420_dev import instrument_seed,noise_trigger_row
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['instrument','noise'],required=True);p.add_argument('--iteration',type=int,default=0);p.add_argument('--index',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
if not 0<=a.iteration<5:raise SystemExit('DEV iteration must be 0..4; five-iteration cap is fixed.')
row=instrument_seed(a.iteration,a.index) if a.mode=='instrument' else noise_trigger_row(a.iteration,a.index)
path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,sort_keys=True)+'\n');os.replace(tmp,path);print(json.dumps(row,indent=2))
