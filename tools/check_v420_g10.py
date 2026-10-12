#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair as r

DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nondress(s):assert not any(int(s) in q for q in DRESS)
def typ(e):return r.expr_type(e)
def key(z,t):return (t,hashlib.blake2b(np.round(np.nan_to_num(z),7).tobytes(),digest_size=12).digest())
def nrmse(z,h):
    z=np.asarray(z,float);h=np.asarray(h,float);A=np.column_stack([np.ones(len(z)),z]);b=np.linalg.lstsq(A,h,rcond=None)[0];p=A@b
    return float(np.sqrt(np.mean((p-h)**2))/max(np.std(h),1e-12))

def tune(e,env,h):return r.tune_constants(e,env,h,np.arange(len(h)),passes=2)
def make_expr(op,args,rng):return r.Expr(op,tuple(args),r._init_param(op,rng) if op in r.PARAM_OPS else None)

def audit(family,seed):
    nondress(seed);env,y,h,meta=r.make_world(family,seed,n=180);support=('x0',) if family=='circular' else ('x0','x1');ref=r.reference_expr(family,meta);referr=nrmse(r.evaluate(ref,env),h);l0=r.l0_hidden_metrics(family,seed)
    class C:
        def __init__(self,e,z):self.e=e;self.z=z;self.d=e.depth();self.n=e.nodes();self.t=typ(e)
    leaves=[C(r.var(k),env[k]) for k in support];allmap={key(c.z,c.t):c for c in leaves};front=leaves[:];counts=[len(allmap)];best=1.0;equiv=None;generated=0
    rng=np.random.default_rng(seed+991)
    def inspect(e,new):
        nonlocal best,equiv,generated
        if e.nodes()>r.MAX_NODES:return
        e=tune(e,env,h);z=r.evaluate(e,env)
        if not np.all(np.isfinite(z)) or np.max(np.abs(np.nan_to_num(z)))>1e8:return
        generated+=1;er=nrmse(z,h);best=min(best,er)
        if er<=1e-8 and equiv is None:equiv=e.text()
        c=C(e,z);k=key(z,c.t)
        if k not in allmap and k not in new:new[k]=c
    for dep in (1,2):
        new={};cum=list(allmap.values())
        for a in front:
            for op in r.UNARY:
                if r._valid_unary(op,a.e):inspect(make_expr(op,(a.e,),rng),new)
        for a in front:
            for b in cum:
                for op in r.BINARY:
                    pairs=((a,b),) if op in ('add','product') else ((a,b),(b,a))
                    for x,z in pairs:
                        if r._valid_binary(op,x.e,z.e):inspect(r.Expr(op,(x.e,z.e),None),new)
        allmap.update(new);front=list(new.values());counts.append(len(allmap))
        if equiv:break
    # depth-3 unary roots: each structure gets continuous constant fitting.
    if not equiv:
        for a in front:
            for op in r.UNARY:
                if not r._valid_unary(op,a.e):continue
                e=tune(make_expr(op,(a.e,),rng),env,h);z=r.evaluate(e,env);generated+=1;er=nrmse(z,h);best=min(best,er)
                if er<=1e-8:equiv=e.text();break
            if equiv:break
    # depth-3 binary roots, exhaustive over depth-2 frontier x all <=2; binary root has no new parameter.
    if not equiv:
        allc=list(allmap.values());T=np.asarray(h,float);T0=T-T.mean();sst=float(T0@T0);BSTEP=96
        for ia in range(0,len(front),48):
            AA=front[ia:ia+48]
            for ib in range(0,len(allc),BSTEP):
                BB=allc[ib:ib+BSTEP];A=np.stack([c.z for c in AA]);B=np.stack([c.z for c in BB]);an=np.asarray([c.n for c in AA]);bn=np.asarray([c.n for c in BB]);nodeok=an[:,None]+bn[None,:]+1<=r.MAX_NODES
                def screen(O,ok):
                    nonlocal best,equiv,generated
                    Z=O-O.mean(2,keepdims=True);cov=np.tensordot(Z,T0,axes=([2],[0]));ss=np.sum(Z*Z,2);rr=np.where((ss>1e-12)&ok,(cov*cov)/(sst*ss),-np.inf);generated+=int(np.sum(ok));mx=float(np.max(rr))
                    if np.isfinite(mx):best=min(best,float(np.sqrt(max(0.,1-mx))))
                    if mx>=1-1e-14:equiv='FUNCTIONALLY_EQUIVALENT_DEPTH3'
                screen(A[:,None,:]+B[None,:,:],nodeok)
                if equiv:break
                screen(A[:,None,:]-B[None,:,:],nodeok)
                if equiv:break
                screen(B[None,:,:]-A[:,None,:],nodeok)
                if equiv:break
                screen(A[:,None,:]*B[None,:,:],nodeok)
                if equiv:break
                boolidx=[j for j,c in enumerate(BB) if c.t=='bool']
                if boolidx:
                    BI=B[boolidx];ok=nodeok[:,boolidx] & np.asarray([[r._valid_binary('interaction',x.e,BB[j].e) for j in boolidx] for x in AA],bool);screen(A[:,None,:]*(BI[None,:,:]>.5),ok)
                if equiv:break
            if equiv:break
    out={'family':family,'seed':seed,'reference_depth':ref.depth(),'reference_nodes':ref.nodes(),'reference_nrmse':referr,'l0_nrmse':l0['nrmse'],'hidden_r2_in_l0':l0['r2'],'unique_functions_by_depth':counts,'screened_candidates':generated,'best_depth_le3_affine_nrmse':best,'depth_le3_equivalent_found':equiv,'pass':bool(ref.depth()>=4 and referr<=.02 and equiv is None and l0['nrmse']>=.20 and l0['r2']<=.50)}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['circular','rank'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();o=audit(a.family,a.seed);pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(o,indent=2,sort_keys=True));print(json.dumps(o,indent=2,sort_keys=True));raise SystemExit(0 if o['pass'] else 2)
if __name__=='__main__':main()
