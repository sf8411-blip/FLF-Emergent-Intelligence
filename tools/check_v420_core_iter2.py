#!/usr/bin/env python3
from __future__ import annotations
import json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair_iter2 as r
from flf_ecf import v420_dev as old
B=540000
DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nd(s):assert not any(int(s) in q for q in DRESS)
def main():
    rows=[]
    def emit(kind,d):
        z={'iteration':2,'kind':kind,**d};rows.append(z);pathlib.Path('results/checkpoints').mkdir(parents=True,exist_ok=True);pathlib.Path('results/checkpoints/v4.20-core-iter2.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');return z
    R2={}
    for j,f in enumerate(('corrupt_circular','corrupt_rank')):
        gains=[]
        for i in range(32):
            s=B+1000+j*200+i;nd(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);gains.append((bb['b0_test_rmse']-bb['b1_test_rmse'])/max(bb['b0_test_rmse'],1e-12))
        ci=old.bootstrap_ci(gains,B=3000,seed=B+j);mu=float(np.mean(gains));R2[f]={'mean':mu,'ci95':ci,'pass':bool(abs(mu)<=.005 and ci[0]<=0<=ci[1])}
    emit('R2',{'families':R2,'pass':all(x['pass'] for x in R2.values())})
    oldgrid=(1.,2.,2*math.pi,3.,5.,9.,.25,.5,.75);vals=[]
    for f,s in (('circular',B+1500),('rank',B+1501)):
        _,_,_,m=r.make_world(f,s);vals.extend(float(x) for x in m.values() if isinstance(x,(int,float)))
    matches=sum(any(abs(v-z)<1e-12 for z in oldgrid) for v in vals);rng=np.random.default_rng(B+1510);exprs=[r.random_expr(('x0','x1'),rng) for _ in range(200)];charge=all(abs(e.mdl_cost()-(e.nodes()+.5*e.constants()))<1e-12 for e in exprs);R3={'continuous_1d_search':True,'old_grid_exact_matches':matches,'constant_charge':r.MDL_CONSTANT_COST,'mdl_formula_ok':charge,'pass':bool(matches==0 and charge and r.MDL_CONSTANT_COST==.5)};emit('R3',R3)
    rb=[]
    for j,f in enumerate(('circular','rank','null','corrupt_circular','corrupt_rank','noise','spurious')):
        s=B+5000+j;nd(s);x=r.run_seed_core(f,s,challenge_B=9,search_budget=8);rb.append({'family':f,'trigger_is_gate':x['trigger_is_gate'],'trigger_open':x['trigger_open'],'search_counts':x['search_counts']})
    RB=all((not x['trigger_is_gate']) and all(int(v)>0 for v in x['search_counts'].values()) for x in rb);emit('R-B',{'rows':rb,'pass':RB})
    rms=[]
    for i in range(64):
        s=B+11000+i;nd(s);env,y,h,m=r.make_world('null',s);bb=r.b0_and_oracle(env,y,h);rms.append(bb['b0_test_rmse'])
    rate=float(np.mean(np.asarray(rms)<=.088));RD={'n':64,'rate_rmse_le_1.10_noise_sd':rate,'median_rmse':float(np.median(rms)),'threshold':.088,'pass':bool(rate>=.95)};emit('R-D',RD)
    ok=all(x['pass'] for x in R2.values()) and R3['pass'] and RB and RD['pass'];out={'iteration':2,'R2':R2,'R3':R3,'R-B':{'rows':rb,'pass':RB},'R-D':RD,'pass':ok};pathlib.Path('results').mkdir(exist_ok=True);pathlib.Path('results/v4.20-core-iter2.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
