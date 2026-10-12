#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair as r
from flf_ecf import v420_dev as old
DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nd(s):assert not any(int(s) in q for q in DRESS)

def main():
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();it=a.iteration
    if not 0<=it<=4:raise SystemExit('iteration must be 0..4')
    B=500000+it*20000;records=[];cp=pathlib.Path(a.checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
    def emit(kind,d):
        z={'iteration':it,'kind':kind,**d};records.append(z);cp.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in records)+'\n');return z
    # R2: exact corruption object used by run_seed_core.
    R2={}
    for j,f in enumerate(('corrupt_circular','corrupt_rank')):
        gains=[]
        for i in range(32):
            s=B+1000+j*200+i;nd(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);gains.append((bb['b0_test_rmse']-bb['b1_test_rmse'])/max(bb['b0_test_rmse'],1e-12))
        ci=old.bootstrap_ci(gains,B=3000,seed=B+j);mu=float(np.mean(gains));R2[f]={'mean':mu,'ci95':ci,'pass':bool(abs(mu)<=.005 and ci[0]<=0<=ci[1])}
    emit('R2',{'families':R2,'pass':all(x['pass'] for x in R2.values())})
    # R3: implementation/used world evidence, no old-grid constants, exact .5 charge.
    oldgrid=(1.,2.,2*math.pi,3.,5.,9.,.25,.5,.75);vals=[]
    for f,s in (('circular',B+1500),('rank',B+1501)):
        nd(s);_,_,_,m=r.make_world(f,s);vals.extend(float(x) for x in m.values() if isinstance(x,(int,float)))
    matches=sum(any(abs(v-z)<1e-12 for z in oldgrid) for v in vals);rng=np.random.default_rng(B+1510);exprs=[r.random_expr(('x0','x1'),rng) for _ in range(200)];charge_ok=all(abs(e.mdl_cost()-(e.nodes()+.5*e.constants()))<1e-12 for e in exprs)
    R3={'continuous_1d_search':True,'old_grid_exact_matches':matches,'constant_charge':r.MDL_CONSTANT_COST,'mdl_formula_ok':charge_ok,'pass':bool(matches==0 and r.MDL_CONSTANT_COST==.5 and charge_ok)};emit('R3',R3)
    # R4: B2b only, equal fixed budget. Success is fraction of oracle gap recovered >=.35.
    R4={};same=[]
    for j,f in enumerate(('circular','rank')):
        succ=[];og=[];counts=[]
        for i in range(32):
            s=B+3000+j*500+i;nd(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);den=bb['b0_test_rmse']-bb['b1_test_rmse'];og.append(den/max(bb['b0_test_rmse'],1e-12));a0,b0,sigs,n=r.select_controls(env,y,bb,s+2001,r.SEARCH_BUDGET);counts.append(n);succ.append((bb['b0_test_rmse']-b0[2]['test_rmse'])/max(den,1e-12)>=.35);same.append(a0[0].text()==b0[0].text())
        rate=float(np.mean(succ));R4[f]={'b2_success_rate':rate,'oracle_ge_.10_rate':float(np.mean(np.asarray(og)>=.10)),'oracle_gain_min':float(np.min(og)),'candidate_eval_min':min(counts),'candidate_eval_max':max(counts),'pass':bool(.10<=rate<=.60 and np.mean(np.asarray(og)>=.10)>=.95 and min(counts)==r.SEARCH_BUDGET and max(counts)==r.SEARCH_BUDGET)}
    emit('R4',{'search_budget':r.SEARCH_BUDGET,'families':R4,'pass':all(x['pass'] for x in R4.values())})
    # R-B exact used function, reduced challenge B because only execution/non-gating is assessed.
    rb=[]
    for j,f in enumerate(('circular','rank','null','corrupt_circular','corrupt_rank','noise','spurious')):
        s=B+5000+j;nd(s);x=r.run_seed_core(f,s,challenge_B=9,search_budget=8);rb.append({'family':f,'trigger_open':x['trigger_open'],'trigger_is_gate':x['trigger_is_gate'],'search_counts':x['search_counts']})
    RB=all((not x['trigger_is_gate']) and all(int(v)>0 for v in x['search_counts'].values()) for x in rb);emit('R-B',{'rows':rb,'pass':RB})
    # R-D real noise-floor property.
    rms=[]
    for i in range(64):
        s=B+11000+i;nd(s);env,y,h,m=r.make_world('null',s);bb=r.b0_and_oracle(env,y,h);rms.append(bb['b0_test_rmse'])
    rate=float(np.mean(np.asarray(rms)<=1.10*.08));RD={'n':64,'rate_rmse_le_1.10_noise_sd':rate,'median_rmse':float(np.median(rms)),'threshold':.088,'pass':bool(rate>=.95)};emit('R-D',RD)
    same_rate=float(np.mean(same));RF={'n':len(same),'same_program_rate':same_rate,'report_as_one_control':bool(same_rate>.90),'pass':True};emit('R-F',RF)
    ok=all(x['pass'] for x in R2.values()) and R3['pass'] and all(x['pass'] for x in R4.values()) and RB and RD['pass']
    out={'iteration':it,'seed_base':B,'search_budget':r.SEARCH_BUDGET,'R2':R2,'R3':R3,'R4':R4,'R-B':{'rows':rb,'pass':RB},'R-D':RD,'R-F':RF,'pass':bool(ok)};pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
