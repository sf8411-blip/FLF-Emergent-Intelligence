#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v420_dress import run_seed,FAMILIES

p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
if not 0<=a.index<32:raise SystemExit('dress index must be 0..31')
row=run_seed(a.family,a.index);path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path);print(json.dumps(row,indent=2,allow_nan=False))
