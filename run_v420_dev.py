#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import flf_ecf.v420_dev as v

def enc(o):
    if hasattr(o,'item'):return o.item()
    raise TypeError(type(o).__name__)

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['instrument','noise'],required=True);p.add_argument('--iteration',type=int,default=0);p.add_argument('--index',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
if not 0<=a.iteration<5:raise SystemExit('DEV iteration must be 0..4; five-iteration cap is fixed.')
# R5 repair after iteration 0: preserve the preregistered 99th percentile,
# block size, score, and false-open cap; increase only Monte Carlo resolution.
if a.iteration>=1:
    _base=v.calibrated_trigger
    def _highres(resid,actions,seed,B=999):
        return _base(resid,actions,seed,B=999)
    v.calibrated_trigger=_highres
# R2 repair after iteration 1: preserve y marginal values exactly while destroying
# Rank mechanism alignment by a large circular shift relative to observations/actions.
# No success metric, threshold, search grammar, or non-corruption world is changed.
if a.iteration>=2:
    _world=v.make_world
    def _shifted_world(family,seed,n=520,action_scale=1.0,difficulty=1.0):
        if family!='corrupt_rank':
            return _world(family,seed,n=n,action_scale=action_scale,difficulty=difficulty)
        env,y,h,meta=_world('rank',seed,n=n,action_scale=action_scale,difficulty=difficulty)
        rr=np.random.default_rng(int(seed)+818181)
        lo=max(1,len(y)//4);hi=max(lo+1,3*len(y)//4)
        shift=int(rr.integers(lo,hi))
        return env,np.roll(y,shift),h,{**meta,'corruption':'circular_shift_y','shift':shift}
    v.make_world=_shifted_world
row=v.instrument_seed(a.iteration,a.index) if a.mode=='instrument' else v.noise_trigger_row(a.iteration,a.index)
path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,sort_keys=True,default=enc)+'\n');os.replace(tmp,path);print(json.dumps(row,indent=2,default=enc))
