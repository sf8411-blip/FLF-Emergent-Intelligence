#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys
from dataclasses import dataclass
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v419_dsl import Expr,var,evaluate,l0_expressions,UNARY,BINARY,valid_unary,valid_binary,expr_type
from flf_ecf.v419_worlds import make_world,hidden_feature,reference_expr

PREREG=hashlib.sha256((ROOT/'docs/v4.19_preregistration.md').read_bytes()).hexdigest()
PREREG_BLOB=os.environ.get('V419_PREREG_BLOB_SHA','UNKNOWN')
DIAG_SEEDS=tuple(range(12000,12012));SEG=20;MAX_NODES=11;SCREEN_R2=1.0-1e-12

@dataclass
class Cand:
    expr: Expr
    out: np.ndarray
    depth: int
    nodes: int
    typ: str

def _hash(a):
    x=np.nan_to_num(a,nan=0.0,posinf=1e8,neginf=-1e8)
    return hashlib.blake2b(np.round(x,8).tobytes(),digest_size=16).digest()
def _key(a,typ):return (_hash(a),typ)
def _segments(a):return [a[i*SEG:(i+1)*SEG] for i in range(len(DIAG_SEEDS))]

def _unary(op,p,a):
    outs=[]
    for x in _segments(a):
        if op=='lag':
            k=int(p);y=np.empty_like(x);y[:k]=x[0];y[k:]=x[:-k]
        elif op=='diff':
            k=int(p);lag=np.empty_like(x);lag[:k]=x[0];lag[k:]=x[:-k];y=x-lag
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
    leaves=[]
    for name in support:
        e=var(name);leaves.append(Cand(e,np.concatenate([z[name] for z in envs]),0,1,'scalar'))
    return envs,np.concatenate(targets),leaves,support

def l0_nrmse(support,envs,target):
    exprs=l0_expressions(support);cols=[np.concatenate([evaluate(e,en) for en in envs]) for e in exprs]
    X=np.c_[np.ones(len(target)),*cols];c=np.linalg.lstsq(X,target,rcond=None)[0];p=X@c
    return float(np.sqrt(np.mean((p-target)**2))/(np.std(target)+1e-12))
def _insert_new(new,allmap,c):
    k=_key(c.out,c.typ);old=allmap.get(k)
    if old is not None:return False
    prior=new.get(k)
    if prior is None or c.nodes<prior.nodes:new[k]=c
    return True

def _build_through_depth2(leaves,target,checkpoint,family):
    allmap={_key(c.out,c.typ):c for c in leaves};frontier=leaves[:];counts=[len(allmap)];generated=0;found=None
    for depth in (1,2):
        new={}
        for a in frontier:
            for op,pars in UNARY:
                if not valid_unary(op,a.expr):continue
                for p in pars:
                    e=Expr(op,(a.expr,),p);nodes=a.nodes+1
                    if nodes>MAX_NODES:continue
                    o=_unary(op,p,a.out)
                    if not np.all(np.isfinite(o)) or np.max(np.abs(o))>1e10:continue
                    c=Cand(e,o,depth,nodes,expr_type(e));generated+=1;_insert_new(new,allmap,c)
                    if _nrmse_affine(o,target)<=1e-8:return allmap,list(new.values()),counts,generated,e
        cumulative=list(allmap.values())
        for a in frontier:
            for b in cumulative:
                for op in BINARY:
                    variants=[(a,b)] if op in ('product','add') else [(a,b),(b,a)]
                    for left,right in variants:
                        if not valid_binary(op,left.expr,right.expr):continue
                        nodes=left.nodes+right.nodes+1
                        if nodes>MAX_NODES:continue
                        e=Expr(op,(left.expr,right.expr));o=_binary(op,left.out,right.out)
                        if not np.all(np.isfinite(o)) or np.max(np.abs(o))>1e10:continue
                        c=Cand(e,o,depth,nodes,expr_type(e));generated+=1;_insert_new(new,allmap,c)
                        if _nrmse_affine(o,target)<=1e-8:return allmap,list(new.values()),counts,generated,e
        allmap.update(new);frontier=list(new.values());counts.append(len(allmap))
        checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp')
        tmp.write_text(json.dumps({'family':family,'depth_completed':depth,'diagnostic_seeds':list(DIAG_SEEDS),'unique_functions_by_depth':counts,'generated_ast_candidates':generated,'found_equivalent':None,'preregistration_sha256':PREREG},indent=2));os.replace(tmp,checkpoint)
    return allmap,frontier,counts,generated,found

def _exact_if_screen(op,a,b,target):
    out=_binary(op,a.out,b.out);return _nrmse_affine(out,target)<=1e-8

def _screen_root_pairs(frontier,allc,target):
    T=np.asarray(target,float);T=T-T.mean();sst=float(T@T);n=len(T);tested=0;best_r2={}
    Braw=np.stack([c.out for c in allc]);Bmean=Braw.mean(1,keepdims=True);Bc=Braw-Bmean
    Bss=np.sum(Bc*Bc,axis=1);Bcov=Bc@T;Bnodes=np.array([c.nodes for c in allc])
    def inspect_matrix(r2,nodeok,op,batch,bpool=allc):
        nonlocal tested
        valid=nodeok & np.isfinite(r2);tested+=int(valid.sum())
        if not np.any(valid):return None
        masked=np.where(valid,r2,-1.0);mx=float(np.max(masked));best_r2[op]=max(best_r2.get(op,-1.0),mx)
        for ii,jj in np.argwhere(masked>=SCREEN_R2):
            a=batch[int(ii)];b=bpool[int(jj)]
            if _exact_if_screen(op,a,b,target):return Expr(op,(a.expr,b.expr))
        return None
    batch_size=192
    for lo in range(0,len(frontier),batch_size):
        batch=frontier[lo:lo+batch_size];Araw=np.stack([c.out for c in batch]);Ac=Araw-Araw.mean(1,keepdims=True)
        Ass=np.sum(Ac*Ac,axis=1);Acov=Ac@T;Anodes=np.array([c.nodes for c in batch]);nodeok=Anodes[:,None]+Bnodes[None,:]+1<=MAX_NODES
        cross=Ac@Bc.T
        with np.errstate(divide='ignore',invalid='ignore'):
            cov=Acov[:,None]+Bcov[None,:];ss=Ass[:,None]+Bss[None,:]+2*cross;r2=(cov*cov)/(sst*ss)
        e=inspect_matrix(r2,nodeok,'add',batch)
        if e:return e,tested,best_r2
        with np.errstate(divide='ignore',invalid='ignore'):
            cov=Acov[:,None]-Bcov[None,:];ss=Ass[:,None]+Bss[None,:]-2*cross;r2=(cov*cov)/(sst*ss)
        e=inspect_matrix(r2,nodeok,'subtract',batch);tested+=int(nodeok.sum())
        if e:return e,tested,best_r2
        M1=Araw@Braw.T;MT=(Araw*T)@Braw.T;M2=(Araw*Araw)@(Braw*Braw).T
        with np.errstate(divide='ignore',invalid='ignore'):
            ss=M2-M1*M1/n;r2=(MT*MT)/(sst*ss)
        e=inspect_matrix(r2,nodeok,'product',batch)
        if e:return e,tested,best_r2
    bool_all=[c for c in allc if c.typ=='bool'];bool_f2=[c for c in frontier if c.typ=='bool']
    def interaction_blocks(Apool,Bpool,label):
        nonlocal tested
        if not Apool or not Bpool:return None
        BB=np.stack([c.out for c in Bpool]);bn=np.array([c.nodes for c in Bpool])
        for lo in range(0,len(Apool),batch_size):
            batch=Apool[lo:lo+batch_size];AA=np.stack([c.out for c in batch]);an=np.array([c.nodes for c in batch]);nodeok=an[:,None]+bn[None,:]+1<=MAX_NODES
            M1=AA@BB.T;MT=(AA*T)@BB.T;M2=(AA*AA)@(BB*BB).T
            with np.errstate(divide='ignore',invalid='ignore'):
                ss=M2-M1*M1/n;r2=(MT*MT)/(sst*ss)
            valid=nodeok & np.isfinite(r2);tested+=int(valid.sum())
            if np.any(valid):
                masked=np.where(valid,r2,-1.0);mx=float(np.max(masked));best_r2[label]=max(best_r2.get(label,-1.0),mx)
                for ii,jj in np.argwhere(masked>=SCREEN_R2):
                    a=batch[int(ii)];b=Bpool[int(jj)]
                    if _exact_if_screen('interaction',a,b,target):return Expr('interaction',(a.expr,b.expr))
        return None
    e=interaction_blocks(frontier,bool_all,'interaction_value_depth2')
    if e:return e,tested,best_r2
    e=interaction_blocks(allc,bool_f2,'interaction_mask_depth2')
    return e,tested,best_r2

def _screen_depth3_unary(frontier,target):
    tested=0;best=1.0
    for a in frontier:
        for op,pars in UNARY:
            if not valid_unary(op,a.expr):continue
            for p in pars:
                if a.nodes+1>MAX_NODES:continue
                e=Expr(op,(a.expr,),p);o=_unary(op,p,a.out);tested+=1;err=_nrmse_affine(o,target);best=min(best,err)
                if err<=1e-8:return e,tested,best
    return None,tested,best

def audit_family(family,checkpoint):
    envs,target,leaves,support=_diagnostics(family);ref=reference_expr(family);refout=np.concatenate([evaluate(ref,en) for en in envs]);referr=_nrmse_affine(refout,target);l0err=l0_nrmse(support,envs,target)
    allmap,frontier,counts,generated,found=_build_through_depth2(leaves,target,checkpoint,family)
    depth3_unary_tested=0;depth3_binary_tested=0;best_unary=None;best_root={}
    if found is None:found,depth3_unary_tested,best_unary=_screen_depth3_unary(frontier,target)
    if found is None:
        allc=list(allmap.values())+frontier;tmp={}
        for c in allc:
            k=_key(c.out,c.typ);old=tmp.get(k)
            if old is None or c.nodes<old.nodes:tmp[k]=c
        found,depth3_binary_tested,best_root=_screen_root_pairs(frontier,list(tmp.values()),target)
    depth3_complete=found is None
    report={'family':family,'diagnostic_seeds':list(DIAG_SEEDS),'operator_support':list(support),'max_depth_exhaustive':3,'max_nodes':MAX_NODES,'unique_functions_through_depth2':counts,'generated_ast_candidates_through_depth2':generated,'depth3_unary_candidates_tested':depth3_unary_tested,'depth3_binary_ast_candidates_tested':depth3_binary_tested,'depth3_best_unary_nrmse':best_unary,'depth3_best_root_r2':best_root,'depth_le3_equivalent_found':found.text() if found else None,'reference_depth':ref.depth(),'reference_nodes':ref.nodes(),'reference_nrmse':referr,'l0_nrmse':l0err,'audit_method':'functional-DP through depth2 + exhaustive vectorized meet-in-the-middle root audit at depth3','pass':depth3_complete and len(counts)==3 and ref.depth()>=4 and ref.nodes()<=MAX_NODES and referr<=.02 and l0err>=.20}
    checkpoint.parent.mkdir(parents=True,exist_ok=True);tmp=checkpoint.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,checkpoint);return report

p=argparse.ArgumentParser();p.add_argument('--family',choices=['circular','rank_interaction','all'],default='all');p.add_argument('--out',default='results/v4.19-reachability.json');p.add_argument('--checkpoint-dir',default='results/reachability-checkpoints');p.add_argument('--code-sha',default=os.environ.get('GITHUB_SHA','UNKNOWN'));a=p.parse_args()
fams=['circular','rank_interaction'] if a.family=='all' else [a.family];results=[audit_family(fam,ROOT/a.checkpoint_dir/f'{fam}.json') for fam in fams]
report={'preregistration_sha256':PREREG,'preregistration_git_blob_sha':PREREG_BLOB,'code_commit_sha':a.code_sha,'diagnostic_count':12,'families':results,'pass':all(x['pass'] for x in results) and len(results)==len(fams)}
out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,out);print(json.dumps(report,indent=2))
