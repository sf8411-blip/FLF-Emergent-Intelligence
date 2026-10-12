#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair as r
from flf_ecf import v420_dev as old
DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nd(s):assert not any(int(s) in q for q in DRESS),s

def core(iteration,base,out,checkpoint):
    rows=[];cp=pathlib.Path(checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
    def emit(kind,d):
        z={'iteration':iteration,'kind':kind,**d};rows.append(z);cp.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');return z
    R2={}
    for j,f in enumerate(('corrupt_circular','corrupt_rank')):
        gains=[]
        for i in range(32):
            s=base+1000+j*200+i;nd(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);gains.append((bb['b0_test_rmse']-bb['b1_test_rmse'])/max(bb['b0_test_rmse'],1e-12))
        ci=old.bootstrap_ci(gains,B=3000,seed=base+j);mu=float(np.mean(gains));R2[f]={'mean':mu,'ci95':ci,'pass':bool(abs(mu)<=.005 and ci[0]<=0<=ci[1])}
    emit('R2',{'families':R2,'pass':all(x['pass'] for x in R2.values())})
    oldgrid=(1.,2.,2*math.pi,3.,5.,9.,.25,.5,.75);vals=[]
    for f,s in (('circular',base+1500),('rank',base+1501)):
        _,_,_,m=r.make_world(f,s);vals.extend(float(x) for x in m.values() if isinstance(x,(int,float)))
    matches=sum(any(abs(v-z)<1e-12 for z in oldgrid) for v in vals);rng=np.random.default_rng(base+1510);exprs=[r.random_expr(('x0','x1'),rng) for _ in range(200)];charge=all(abs(e.mdl_cost()-(e.nodes()+.5*e.constants()))<1e-12 for e in exprs);R3={'continuous_1d_search':True,'old_grid_exact_matches':matches,'constant_charge':r.MDL_CONSTANT_COST,'mdl_formula_ok':charge,'pass':bool(matches==0 and charge and r.MDL_CONSTANT_COST==.5)};emit('R3',R3)
    rb=[]
    for j,f in enumerate(('circular','rank','null','corrupt_circular','corrupt_rank','noise','spurious')):
        s=base+5000+j;nd(s);x=r.run_seed_core(f,s,challenge_B=9,search_budget=8);rb.append({'family':f,'trigger_is_gate':x['trigger_is_gate'],'trigger_open':x['trigger_open'],'search_counts':x['search_counts']})
    RB=all((not x['trigger_is_gate']) and all(int(v)>0 for v in x['search_counts'].values()) for x in rb);emit('R-B',{'rows':rb,'pass':RB})
    rms=[]
    for i in range(64):
        s=base+11000+i;nd(s);env,y,h,m=r.make_world('null',s);bb=r.b0_and_oracle(env,y,h);rms.append(bb['b0_test_rmse'])
    rate=float(np.mean(np.asarray(rms)<=.088));RD={'n':64,'rate_rmse_le_1.10_noise_sd':rate,'median_rmse':float(np.median(rms)),'threshold':.088,'pass':bool(rate>=.95)};emit('R-D',RD)
    ok=all(x['pass'] for x in R2.values()) and R3['pass'] and RB and RD['pass'];res={'iteration':iteration,'seed_base':base,'R2':R2,'R3':R3,'R-B':{'rows':rb,'pass':RB},'R-D':RD,'pass':ok};pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(res,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)

def r4batch(iteration,base,family,batch,out):
    if not 0<=batch<4:raise SystemExit('batch 0..3')
    fb=base+3000+(0 if family=='circular' else 500);rows=[]
    for i in range(8):
        s=fb+batch*8+i;nd(s);env,y,h,m=r.make_world(family,s);bb=r.b0_and_oracle(env,y,h);den=bb['b0_test_rmse']-bb['b1_test_rmse'];og=den/max(bb['b0_test_rmse'],1e-12);aa,b2,sigs,n=r.select_controls(env,y,bb,s+2001,r.SEARCH_BUDGET);frac=(bb['b0_test_rmse']-b2[2]['test_rmse'])/max(den,1e-12);rows.append({'iteration':iteration,'family':family,'seed':s,'oracle_gain':float(og),'b2_fraction':float(frac),'success':bool(frac>=.35),'candidate_evals':int(n),'same_program':aa[0].text()==b2[0].text()})
    q=pathlib.Path(out);q.parent.mkdir(parents=True,exist_ok=True);q.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n')

def spuriousbatch(iteration,base,batch,out):
    if not 0<=batch<30:raise SystemExit('batch 0..29')
    rows=[];start=base+7000+batch*10
    for i in range(10):
        s=start+i;nd(s);env,y,h,m=r.make_world('spurious',s);bb=r.b0_and_oracle(env,y,h);sel,n=r.select_guided(env,y,bb,s+1001,r.SEARCH_BUDGET,False);_,e,z,fit,_=sel;ok,r2,gain=r.prechallenge(e,z,fit,bb);row={'iteration':iteration,'seed':s,'prechallenge':bool(ok),'novelty_r2':float(r2),'validation_gain':float(gain),'program':e.text(),'candidate_evals':int(n),'rho':float(m['rho'])}
        if ok:
            pv,cg,d=r.challenge(e,'spurious',m,s+700001,B=999);row.update({'challenge_p':float(pv),'challenge_gain':float(cg),'acf_length':int(d['acf_length']),'block_length':int(d['block_length']),'false_pass':bool(pv<=r.CHALLENGE_P_MAX)})
        rows.append(row)
    q=pathlib.Path(out);q.parent.mkdir(parents=True,exist_ok=True);q.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n')

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='mode',required=True)
    q=sp.add_parser('core');q.add_argument('--iteration',type=int,required=True);q.add_argument('--base',type=int,required=True);q.add_argument('--out',required=True);q.add_argument('--checkpoint',required=True)
    q=sp.add_parser('r4');q.add_argument('--iteration',type=int,required=True);q.add_argument('--base',type=int,required=True);q.add_argument('--family',choices=['circular','rank'],required=True);q.add_argument('--batch',type=int,required=True);q.add_argument('--out',required=True)
    q=sp.add_parser('spurious');q.add_argument('--iteration',type=int,required=True);q.add_argument('--base',type=int,required=True);q.add_argument('--batch',type=int,required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.mode=='core':core(a.iteration,a.base,a.out,a.checkpoint)
    elif a.mode=='r4':r4batch(a.iteration,a.base,a.family,a.batch,a.out)
    else:spuriousbatch(a.iteration,a.base,a.batch,a.out)
if __name__=='__main__':main()
