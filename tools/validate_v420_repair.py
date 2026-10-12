#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair as r
from flf_ecf import v420_dev as old

DRESS=[range(320000,320032),range(321000,321032),range(322000,322032),range(323000,323032),range(324000,324032),range(325000,325032),range(326000,326032)]
def nondress(s):
    assert not any(int(s) in q for q in DRESS),s

def cp_upper(x,n,alpha=.05):
    if n<=0:return 1.0
    if x>=n:return 1.0
    lo,hi=0.,1.
    for _ in range(80):
        m=(lo+hi)/2
        c=sum(math.comb(n,k)*(m**k)*((1-m)**(n-k)) for k in range(x+1))
        if c>alpha:lo=m
        else:hi=m
    return (lo+hi)/2

def nrmse_affine(z,h):
    z=np.asarray(z,float);h=np.asarray(h,float);A=np.column_stack([np.ones(len(z)),z]);b=np.linalg.lstsq(A,h,rcond=None)[0];p=A@b
    return float(np.sqrt(np.mean((p-h)**2))/max(np.std(h),1e-12))
def fhash(z,typ):return (typ,hashlib.blake2b(np.round(np.nan_to_num(z),7).tobytes(),digest_size=12).digest())

def g10_family(family,seed):
    nondress(seed);env,y,h,meta=r.make_world(family,seed,n=220);support=('x0',) if family=='circular' else ('x0','x1');idx=np.arange(len(h));
    ref=r.reference_expr(family,meta);ref_err=nrmse_affine(r.evaluate(ref,env),h);l0=r.l0_hidden_metrics(family,seed)
    @dataclass_local
    class C:pass

def dataclass_local(cls):
    # tiny mutable record without importing dataclasses again
    return cls

def _cand(expr,out):
    class C:pass
    c=C();c.expr=expr;c.out=np.asarray(out,float);c.depth=expr.depth();c.nodes=expr.nodes();c.typ=r.expr_type(expr);return c

def _g10(family,seed):
    nondress(seed);env,y,h,meta=r.make_world(family,seed,n=220);support=('x0',) if family=='circular' else ('x0','x1');idx=np.arange(len(h));ref=r.reference_expr(family,meta);ref_err=nrmse_affine(r.evaluate(ref,env),h);l0=r.l0_hidden_metrics(family,seed)
    leaves=[_cand(r.var(k),env[k]) for k in support];allmap={fhash(c.out,c.typ):c for c in leaves};frontier=leaves[:];counts=[len(allmap)];generated=0;best=1e9;equiv=False
    def add(e,new):
        nonlocal generated,best,equiv
        if e.nodes()>r.MAX_NODES:return
        e=r.tune_constants(e,env,h,idx,passes=2);z=r.evaluate(e,env)
        if not np.all(np.isfinite(z)) or np.max(np.abs(np.nan_to_num(z)))>1e8:return
        generated+=1;er=nrmse_affine(z,h);best=min(best,er);equiv=equiv or er<=1e-8;c=_cand(e,z);k=fhash(z,c.typ)
        if k not in allmap and k not in new:new[k]=c
    # exhaustive structural construction through depth 2; fitted constants on every structure.
    for dep in (1,2):
        new={};cumulative=list(allmap.values())
        for a in frontier:
            for op in r.UNARY:
                if not r._valid_unary(op,a.expr):continue
                add(r.Expr(op,(a.expr,),r._init_param(op,np.random.default_rng(12345+generated)) if op in r.PARAM_OPS else None),new)
        for a in frontier:
            for b in cumulative:
                for op in r.BINARY:
                    variants=((a,b),) if op in ('product','add') else ((a,b),(b,a))
                    for x,z in variants:
                        if r._valid_binary(op,x.expr,z.expr):add(r.Expr(op,(x.expr,z.expr),None),new)
        allmap.update(new);frontier=list(new.values());counts.append(len(allmap))
    # depth-3 unary exhaustive screen with continuous fitting.
    for a in frontier:
        for op in r.UNARY:
            if not r._valid_unary(op,a.expr):continue
            e=r.Expr(op,(a.expr,),r._init_param(op,np.random.default_rng(777+generated)) if op in r.PARAM_OPS else None);e=r.tune_constants(e,env,h,idx,passes=2);z=r.evaluate(e,env);generated+=1;best=min(best,nrmse_affine(z,h));equiv=equiv or nrmse_affine(z,h)<=1e-8
    # exhaustive binary-root screen at depth 3, vectorized in batches. Binary roots have no new constants.
    allc=list(allmap.values());B=np.stack([c.out for c in allc]);Bn=np.asarray([c.nodes for c in allc]);Btyp=[c.typ for c in allc];batch=96
    T=np.asarray(h,float);T0=T-T.mean();sst=float(T0@T0);n=len(T)
    def err_from(out):
        nonlocal best,equiv
        # out shape m,k,n
        mean=out.mean(axis=2,keepdims=True);zc=out-mean;cov=np.tensordot(zc,T0,axes=([2],[0]));ss=np.sum(zc*zc,axis=2);r2=np.where(ss>1e-12,(cov*cov)/(sst*ss),0.0);err=np.sqrt(np.maximum(0,1-r2));m=float(np.nanmin(err));best=min(best,m);equiv=equiv or m<=1e-8
    for lo in range(0,len(frontier),batch):
        aa=frontier[lo:lo+batch];A=np.stack([c.out for c in aa]);An=np.asarray([c.nodes for c in aa]);nodeok=An[:,None]+Bn[None,:]+1<=r.MAX_NODES
        # add, subtract, product; invalid oversized cells are replaced with constant and ignored via +inf after direct r2 calculation.
        for op in ('add','subtract','product'):
            if op=='add':O=A[:,None,:]+B[None,:,:]
            elif op=='subtract':O=A[:,None,:]-B[None,:,:]
            else:O=A[:,None,:]*B[None,:,:]
            M=O.mean(2,keepdims=True);Z=O-M;cov=np.tensordot(Z,T0,axes=([2],[0]));ss=np.sum(Z*Z,2);rr=np.where((ss>1e-12)&nodeok,(cov*cov)/(sst*ss),-np.inf);mx=float(np.nanmax(rr));
            if np.isfinite(mx):best=min(best,float(np.sqrt(max(0,1-mx))));equiv=equiv or mx>=1-1e-14
        bool_idx=np.asarray([i for i,t in enumerate(Btyp) if t=='bool'],int)
        if len(bool_idx):
            BB=B[bool_idx];BN=Bn[bool_idx];ok=An[:,None]+BN[None,:]+1<=r.MAX_NODES;O=A[:,None,:]*(BB[None,:,:]>.5);M=O.mean(2,keepdims=True);Z=O-M;cov=np.tensordot(Z,T0,axes=([2],[0]));ss=np.sum(Z*Z,2);rr=np.where((ss>1e-12)&ok,(cov*cov)/(sst*ss),-np.inf);mx=float(np.nanmax(rr));
            if np.isfinite(mx):best=min(best,float(np.sqrt(max(0,1-mx))));equiv=equiv or mx>=1-1e-14
    return {'family':family,'reference_depth':ref.depth(),'reference_nodes':ref.nodes(),'reference_nrmse':ref_err,'l0_nrmse':l0['nrmse'],'hidden_r2_in_l0':l0['r2'],'unique_functions_by_depth':counts,'generated_or_screened':generated,'best_depth_le3_affine_nrmse':best,'depth_le3_equivalent_found':bool(equiv),'pass':bool(ref.depth()>=4 and ref_err<=.02 and not equiv and l0['nrmse']>=.20 and l0['r2']<=.50)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--iteration',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--checkpoint',required=True);a=ap.parse_args();it=a.iteration
    if not 0<=it<=4:raise SystemExit('R-A..R-F repair iterations are limited to 0..4')
    B=500000+it*20000;rows=[]
    cp=pathlib.Path(a.checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint(kind,data):
        rec={'iteration':it,'kind':kind,**data};rows.append(rec);cp.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n')
    # R-A structure + G10 + L0 novelty.
    rng=np.random.default_rng(B+1);clean=all(r.structure_is_clean(r.random_expr(('x0','x1','u0','u1'),rng)) for _ in range(1000));g10=[_g10('circular',B+10),_g10('rank',B+11)];RA=bool(clean and all(x['pass'] for x in g10));checkpoint('R-A',{'clean_structure_sampler':clean,'g10':g10,'pass':RA})
    # R2/R3 carried forward on the actual corruption/world objects.
    r2={};r3_meta=[]
    for j,f in enumerate(('corrupt_circular','corrupt_rank')):
        gains=[]
        for i in range(32):
            s=B+1000+j*200+i;nondress(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);gains.append((bb['b0_test_rmse']-bb['b1_test_rmse'])/max(bb['b0_test_rmse'],1e-12))
        lo,hi=old.bootstrap_ci(gains,B=2000,seed=B+j);r2[f]={'mean':float(np.mean(gains)),'ci95':[lo,hi],'pass':bool(abs(np.mean(gains))<=.005 and lo<=0<=hi)}
    for f,s in [('circular',B+1500),('rank',B+1501)]:
        _,_,_,m=r.make_world(f,s);r3_meta.append(m)
    oldgrid=(1.,2.,2*math.pi,3.,5.,9.,.25,.5,.75);gridmatch=sum(any(abs(float(v)-z)<1e-12 for z in oldgrid) for m in r3_meta for v in m.values() if isinstance(v,(int,float)))
    R3={'continuous_1d_search':True,'mdl_constant_cost':r.MDL_CONSTANT_COST,'old_grid_exact_matches':int(gridmatch),'pass':bool(r.MDL_CONSTANT_COST==.5 and gridmatch==0)};checkpoint('R2-R3',{'R2':r2,'R3':R3,'pass':bool(all(x['pass'] for x in r2.values()) and R3['pass'])})
    # R4 calibration: B2b only, 32 fresh seeds/family, fixed iteration budget.
    R4={};rf_same=[]
    for j,f in enumerate(('circular','rank')):
        succ=[];og=[]
        for i in range(32):
            s=B+3000+j*500+i;nondress(s);env,y,h,m=r.make_world(f,s);bb=r.b0_and_oracle(env,y,h);den=bb['b0_test_rmse']-bb['b1_test_rmse'];og.append(den/max(bb['b0_test_rmse'],1e-12));a0,b0,sigs,n=r.select_controls(env,y,bb,s+2001,r.SEARCH_BUDGET);gap=(bb['b0_test_rmse']-b0[2]['test_rmse'])/max(den,1e-12);succ.append(gap>=.35);rf_same.append(a0[0].text()==b0[0].text())
        rate=float(np.mean(succ));R4[f]={'b2_success_rate':rate,'oracle_gain_min':float(np.min(og)),'oracle_ge_.10_rate':float(np.mean(np.asarray(og)>=.10)),'pass':bool(.10<=rate<=.60 and np.mean(np.asarray(og)>=.10)>=.95)}
    checkpoint('R4',{'search_budget':r.SEARCH_BUDGET,'families':R4,'pass':all(x['pass'] for x in R4.values())})
    # R-B: non-gating exact run_seed_core path, even when diagnostic trigger is closed.
    rb=[]
    for j,f in enumerate(('circular','rank','null','corrupt_circular','corrupt_rank','noise')):
        s=B+5000+j;nondress(s);x=r.run_seed_core(f,s,challenge_B=19);rb.append({'family':f,'trigger_open':x['trigger_open'],'trigger_is_gate':x['trigger_is_gate'],'search_counts':x['search_counts']})
    RB=all((not x['trigger_is_gate']) and all(v>0 for v in x['search_counts'].values()) for x in rb);checkpoint('R-B',{'rows':rb,'pass':RB})
    # R-C: hard spurious controls. One best guided candidate/seed; collect >=30 passing novelty+gain within <=300.
    pre=[];checked=0
    for i in range(300):
        s=B+7000+i;nondress(s);env,y,h,m=r.make_world('spurious',s);bb=r.b0_and_oracle(env,y,h);sel,n=r.select_guided(env,y,bb,s+1001,r.SEARCH_BUDGET,False);_,e,z,fit,_=sel;ok,r2v,gv=r.prechallenge(e,z,fit,bb);checked+=1
        if ok:
            pv,cg,diag=r.challenge(e,'spurious',m,s+700001,B=199);pre.append({'seed':s,'novelty_r2':r2v,'validation_gain':gv,'challenge_p':pv,'challenge_gain':cg,'acf_length':diag['acf_length'],'block_length':diag['block_length'],'false_pass':pv<=r.CHALLENGE_P_MAX})
            if len(pre)>=30:break
    fp=sum(x['false_pass'] for x in pre);up=cp_upper(fp,len(pre));block_ok=all(x['block_length']>=x['acf_length'] for x in pre);RC=bool(len(pre)>=30 and fp/len(pre)<=.10 and up<=.10 and block_ok);checkpoint('R-C',{'searched_seeds':checked,'prechallenge_count':len(pre),'false_passes':fp,'false_pass_rate':fp/max(len(pre),1),'cp95_upper':up,'block_ge_acf':block_ok,'examples':pre[:10],'pass':RC})
    # R-D real Null noise-floor property on 64 seeds.
    vals=[]
    for i in range(64):
        s=B+11000+i;nondress(s);env,y,h,m=r.make_world('null',s);bb=r.b0_and_oracle(env,y,h);vals.append(bb['b0_test_rmse'])
    rate=float(np.mean(np.asarray(vals)<=1.10*.08));RD=rate>=.95;checkpoint('R-D',{'n':64,'rate_rmse_le_1.10_noise_sd':rate,'median_rmse':float(np.median(vals)),'threshold':.088,'pass':RD})
    # R-F same-control diagnostic on the B2 calibration bank.
    same=float(np.mean(rf_same));checkpoint('R-F',{'n':len(rf_same),'same_program_rate':same,'report_as_one_control':bool(same>.90),'pass':True})
    # R-E is enforced by executable tests/CI, not a self-declared flag.
    overall=bool(RA and all(x['pass'] for x in r2.values()) and R3['pass'] and all(x['pass'] for x in R4.values()) and RB and RC and RD)
    out={'iteration':it,'search_budget':r.SEARCH_BUDGET,'R-A':rows[0],'R2-R3':rows[1],'R4':rows[2],'R-B':rows[3],'R-C':rows[4],'R-D':rows[5],'R-F':rows[6],'R-E':'REQUIRES_CI','instrument_valid_without_RE':overall,'instrument_valid':False,'note':'instrument_valid becomes true only when same-HEAD R-E end-to-end CI passes.'}
    pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
    if not overall:raise SystemExit(2)
if __name__=='__main__':main()
