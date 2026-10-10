#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys
from dataclasses import dataclass
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v419_dsl import Expr,var,evaluate,l0_expressions,UNARY,BINARY,valid_unary,valid_binary
from flf_ecf.v419_worlds import make_world,hidden_feature,reference_expr

PREREG=hashlib.sha256((ROOT/'docs/v4.19_preregistration.md').read_bytes()).hexdigest()
PREREG_BLOB=os.environ.get('V419_PREREG_BLOB_SHA','UNKNOWN')
DIAG_SEEDS=tuple(range(8000,8012))
SEG=20

@dataclass
class Cand:
    expr: Expr
    out: np.ndarray
    depth: int


def _hash(a):
    x=np.nan_to_num(a,nan=0.0,posinf=1e8,neginf=-1e8)
    return hashlib.blake2b(np.round(x,8).tobytes(),digest_size=16).digest()

def _segments(a): return [a[i*SEG:(i+1)*SEG] for i in range(len(DIAG_SEEDS))]

def _unary(op,p,a):
    outs=[]
    for x in _segments(a):
        if op=='lag':
            k=int(p); y=np.empty_like(x);y[:k]=x[0];y[k:]=x[:-k]
        elif op=='diff':
            k=int(p); lag=np.empty_like(x);lag[:k]=x[0];lag[k:]=x[:-k];y=x-lag
        elif op=='cumsum':y=np.cumsum(x)
        elif op=='filter_mean':
            w=int(p);y=np.empty_like(x);cs=np.cumsum(np.r_[0.0,x])
            for i in range(len(x)):
                j=max(0,i-w+1);y[i]=(cs[i+1]-cs[j])/(i-j+1)
        elif op=='rank01':
            w=int(p);y=np.empty_like(x)
            for i in range(len(x)):
                j=max(0,i-w+1);h=x[j:i+1];y[i]=(np.sum(h<x[i])+.5*np.sum(h==x[i]))/len(h)
        elif op=='threshold_gt':y=(x>float(p)).astype(float)
        elif op=='threshold_lt':y=(x<float(p)).astype(float)
        elif op=='modulo':y=np.mod(x,float(p))
        elif op=='negate':y=-x
        elif op=='abs':y=np.abs(x)
        else:raise ValueError(op)
        outs.append(y)
    return np.concatenate(outs)

def _binary(op,a,b):
    if op=='product':return a*b
    if op=='interaction':return a*(b>.5)
    if op=='add':return a+b
    if op=='subtract':return a-b
    raise ValueError(op)

def _nrmse_affine(x,y):
    X=np.c_[np.ones(len(x)),x];c=np.linalg.lstsq(X,y,rcond=None)[0];p=X@c
    return float(np.sqrt(np.mean((p-y)**2))/(np.std(y)+1e-12))

def _diagnostics(family):
    envs=[];targets=[];support=('x0',) if family=='circular' else ('x0','x1')
    for s in DIAG_SEEDS:
        env,u,y=make_world(family,s,SEG);envs.append(env);targets.append(hidden_feature(family,env))
    leaves=[Cand(var(name),np.concatenate([e[name] for e in envs]),0) for name in support]
    return envs,np.concatenate(targets),leaves

def l0_nrmse(family,envs,target):
    support=('x0',) if family=='circular' else ('x0','x1');exprs=l0_expressions(support)
    cols=[np.concatenate([evaluate(e,en) for en in envs]) for e in exprs]
    X=np.c_[np.ones(len(target)),*cols];c=np.linalg.lstsq(X,target,rcond=None)[0];p=X@c
    return float(np.sqrt(np.mean((p-target)**2))/(np.std(target)+1e-12))

def audit_family(family, checkpoint):
    envs,target,leaves=_diagnostics(family)
    ref=reference_expr(family);refout=np.concatenate([evaluate(ref,en) for en in envs]);referr=_nrmse_affine(refout,target);l0err=l0_nrmse(family,envs,target)
    allmap={_hash(c.out):c for c in leaves};frontier=leaves[:];counts=[len(allmap)];found=None;generated=0
    for depth in (1,2,3):
        new={}
        for a in frontier:
            for op,pars in UNARY:
                if not valid_unary(op,a.expr):continue
                for p in pars:
                    try:o=_unary(op,p,a.out)
                    except Exception:continue
                    if not np.all(np.isfinite(o)) or np.max(np.abs(o))>1e10:continue
                    k=_hash(o)
                    if k in allmap or k in new:continue
                    e=Expr(op,(a.expr,),p);new[k]=Cand(e,o,depth);generated+=1
                    if _nrmse_affine(o,target)<=1e-8:found=e;break
                if found:break
            if found:break
        if found:break
        cumulative=list(allmap.values())
        for a in frontier:
            if found:break
            for b in cumulative:
                for op in BINARY:
                    variants=[(a,b)] if op in ('product','add') else [(a,b),(b,a)]
                    for left,right in variants:
                        if not valid_binary(op,left.expr,right.expr):continue
                        if op in ('product','add') and left.expr.text()>right.expr.text():continue
                        try:o=_binary(op,left.out,right.out)
                        except Exception:continue
                        if not np.all(np.isfinite(o)) or np.max(np.abs(o))>1e10:continue
                        k=_hash(o)
                        if k in allmap or k in new:continue
                        e=Expr(op,(left.expr,right.expr));new[k]=Cand(e,o,depth);generated+=1
                        if _nrmse_affine(o,target)<=1e-8:found=e;break
                    if found:break
                if found:break
        allmap.update(new);frontier=list(new.values());counts.append(len(allmap))
        checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp')
        tmp.write_text(json.dumps({'family':family,'depth_completed':depth,'unique_functions':counts,'generated':generated,'found_equivalent':found.text() if found else None,'preregistration_sha256':PREREG},indent=2));os.replace(tmp,checkpoint)
    return {'family':family,'diagnostic_seeds':list(DIAG_SEEDS),'max_depth_exhaustive':3,'unique_functions_by_depth':counts,'generated_candidates':generated,
            'depth_le3_equivalent_found':found.text() if found else None,'reference_depth':ref.depth(),'reference_nrmse':referr,'l0_nrmse':l0err,
            'pass':found is None and len(counts)==4 and ref.depth()>=4 and referr<=.02 and l0err>=.20}

p=argparse.ArgumentParser();p.add_argument('--family',choices=['circular','rank_interaction','all'],default='all');p.add_argument('--out',default='results/v4.19-reachability.json');p.add_argument('--checkpoint-dir',default='results/reachability-checkpoints');p.add_argument('--code-sha',default=os.environ.get('GITHUB_SHA','UNKNOWN'));a=p.parse_args()
fams=['circular','rank_interaction'] if a.family=='all' else [a.family];results=[]
for fam in fams:results.append(audit_family(fam,ROOT/a.checkpoint_dir/f'{fam}.json'))
report={'preregistration_sha256':PREREG,'preregistration_git_blob_sha':PREREG_BLOB,'code_commit_sha':a.code_sha,'diagnostic_count':12,'families':results,'pass':len(results)==2 and all(x['pass'] for x in results)}
out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,out);print(json.dumps(report,indent=2))
