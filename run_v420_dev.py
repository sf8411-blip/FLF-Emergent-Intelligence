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
    _base_trigger=v.calibrated_trigger
    def _highres(resid,actions,seed,B=999):
        return _base_trigger(resid,actions,seed,B=999)
    v.calibrated_trigger=_highres
# R2 repair after iteration 1/2: corruption must preserve the exact y marginal
# while destroying alignment to x/actions. From iteration 3 onward BOTH corrupt
# families use a large seed-determined circular shift, permitted by R2.
if a.iteration>=2:
    _world=v.make_world
    def _shifted_world(family,seed,n=520,action_scale=1.0,difficulty=1.0):
        if family not in ('corrupt_rank','corrupt_circular'):
            return _world(family,seed,n=n,action_scale=action_scale,difficulty=difficulty)
        base_family='rank' if family=='corrupt_rank' else 'circular'
        env,y,h,meta=_world(base_family,seed,n=n,action_scale=action_scale,difficulty=difficulty)
        if family=='corrupt_circular' and a.iteration<3:
            # iteration 2 retained the original block-permutation Circular control.
            rr=np.random.default_rng(int(seed));return env,v.block_permute(y,rr,20),h,{**meta,'corruption':'block_permute_y'}
        rr=np.random.default_rng(int(seed)+818181)
        lo=max(1,len(y)//4);hi=max(lo+1,3*len(y)//4);shift=int(rr.integers(lo,hi))
        return env,np.roll(y,shift),h,{**meta,'corruption':'circular_shift_y','shift':shift}
    v.make_world=_shifted_world
# R2 oracle-instrument repair at iteration 3: the oracle extension is not forced
# to use an irrelevant feature. Its scalar coefficient is selected on validation
# from a fixed shrinkage path that includes beta=0. This removes finite-sample
# null overfit while retaining the same hidden feature and R4 thresholds.
if a.iteration>=3:
    _base_oracle=v.b0_and_oracle
    def _gated_oracle(env,y,h):
        b=_base_oracle(env,y,h)
        raw=v.extension_fit(b,y,h);beta=float(raw['beta']);zz=np.asarray(raw['z'])
        candidates=[]
        for scale in (0.0,0.25,0.5,0.75,1.0):
            pp=b['pred']+(scale*beta)*zz
            candidates.append((v.rmse(y[b['val']],pp[b['val']]),scale,pp))
        _,scale,pp=min(candidates,key=lambda z:z[0])
        o=dict(raw);o['beta']=scale*beta;o['pred']=pp;o['val_rmse']=v.rmse(y[b['val']],pp[b['val']]);o['test_rmse']=v.rmse(y[b['te']],pp[b['te']]);o['validation_scale']=scale
        b['oracle']=o;b['b1_test_rmse']=o['test_rmse'];return b
    v.b0_and_oracle=_gated_oracle
# Final allowed DEV iteration: satisfy R3 literally with bounded continuous 1-D
# refinement (not a fixed constant grid), and validate R6's fresh-trace challenge
# on the same latent world parameters at a different intervention amplitude.
if a.iteration>=4:
    from flf_ecf import v420_protocol as protocol
    v.tune_constants=protocol.tune_constants_continuous
    _instrument_seed=v.instrument_seed
    def _final_instrument_seed(iteration,index):
        row=_instrument_seed(iteration,index)
        row['R3_constant_search']='continuous_1d_refinement'
        b=120000+int(iteration)*10000;i=int(index)
        for fam in ('circular','rank'):
            env,y,h,meta=v.make_world(fam,b+11000+i)
            prog=protocol.oracle_program(fam,meta)
            pv,gain,meta2=protocol.challenge_same_world(prog,fam,meta,b+920000+i,B=999,action_scale=1.6)
            row[f'R6_{fam}_positive_challenge_p']=pv
            row[f'R6_{fam}_positive_challenge_gain']=gain
            row[f'R6_{fam}_same_meta']=bool(meta==meta2)
        return row
    v.instrument_seed=_final_instrument_seed
row=v.instrument_seed(a.iteration,a.index) if a.mode=='instrument' else v.noise_trigger_row(a.iteration,a.index)
path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,sort_keys=True,default=enc)+'\n');os.replace(tmp,path);print(json.dumps(row,indent=2,default=enc))
