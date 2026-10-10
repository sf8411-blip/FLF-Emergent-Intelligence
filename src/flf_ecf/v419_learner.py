from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .v419_dsl import Expr, evaluate, l0_expressions, mutate_expr, mdl_cost, var

# IMPORTANT: this module must not import benchmark generators or oracle code.

@dataclass
class Fit:
    coef: np.ndarray
    pred: np.ndarray
    residual: np.ndarray
    rmse: float
    dl: float


def _matrix(exprs:list[Expr], env:dict[str,np.ndarray]) -> np.ndarray:
    cols=[]
    for e in exprs:
        z=np.asarray(evaluate(e,env),float)
        z=np.nan_to_num(z,nan=0.0,posinf=1e6,neginf=-1e6)
        scale=np.std(z)+1e-9; cols.append((z-np.mean(z))/scale)
    return np.c_[np.ones(len(next(iter(env.values())))), *cols]


def linear_fit(exprs:list[Expr], env:dict[str,np.ndarray], target:np.ndarray, ridge:float=1e-3) -> Fit:
    X=_matrix(exprs,env); y=np.asarray(target,float)
    A=X.T@X + ridge*np.eye(X.shape[1]); b=X.T@y
    coef=np.linalg.solve(A,b); pred=X@coef; res=y-pred
    rmse=float(np.sqrt(np.mean(res*res)))
    var=max(float(np.var(res)),1e-12)
    dl=0.5*len(y)*math.log(var)+0.5*X.shape[1]*math.log(max(len(y),2))
    return Fit(coef,pred,res,rmse,dl)


def apply_fit(exprs:list[Expr], fit:Fit, env:dict[str,np.ndarray], target:np.ndarray) -> tuple[float,np.ndarray]:
    X=_matrix(exprs,env); pred=X@fit.coef; res=np.asarray(target)-pred
    return float(np.sqrt(np.mean(res*res))),res


def _acf1(x:np.ndarray)->float:
    x=x-np.mean(x)
    if len(x)<3 or np.std(x)<1e-12:return 0.0
    return float(np.corrcoef(x[:-1],x[1:])[0,1])


def trigger_components(residual:np.ndarray, actions:np.ndarray, base_dl:float) -> tuple[float,float,float,float]:
    n=len(residual); r=np.asarray(residual,float)
    ac=abs(_acf1(r))*math.sqrt(max(n,1))
    slopes=[]
    for j in range(actions.shape[1]):
        a=actions[:,j]
        slopes.append(abs(float(np.corrcoef(a,r)[0,1])) if np.std(a)>1e-12 and np.std(r)>1e-12 else 0.0)
    inc=max(slopes,default=0.0)*math.sqrt(max(n,1))
    sign_changes=np.mean(np.sign(r[1:])!=np.sign(r[:-1])) if n>1 else .5
    comp=max(0.0,(0.5-sign_changes))*8.0
    score=ac+inc+comp
    return score,ac,inc,comp

@dataclass
class SearchResult:
    accepted: bool
    expr: Expr|None
    evaluations: int
    base_rmse: float
    train_rmse: float
    validation_rmse: float
    mdl_improvement: float
    trigger_score: float
    trace_signature: list[float]|None


def synthesize_operator(train_env:dict[str,np.ndarray], train_y:np.ndarray,
                        val_env:dict[str,np.ndarray], val_y:np.ndarray,
                        actions:np.ndarray, budget:int, seed:int,
                        shuffled_residual:bool=False, trigger_enabled:bool=True,
                        max_depth:int=6, max_nodes:int=11) -> SearchResult:
    variables=tuple(sorted(train_env))
    base_exprs=l0_expressions(variables)
    base=linear_fit(base_exprs,train_env,train_y)
    base_val,_=apply_fit(base_exprs,base,val_env,val_y)
    residual=base.residual.copy()
    rng=np.random.default_rng(seed)
    if shuffled_residual: rng.shuffle(residual)
    trig,*_=trigger_components(residual,actions,base.dl)
    if (not trigger_enabled) or trig<2.5:
        return SearchResult(False,None,0,base.rmse,base.rmse,base_val,0.0,trig,None)

    pool=[var(v) for v in variables]
    scored=[]; seen=set(); evals=0
    best=None; best_score=-1e99
    while evals<budget:
        e=mutate_expr(pool,variables,rng,max_depth,max_nodes)
        try:z=np.asarray(evaluate(e,train_env),float)
        except Exception: continue
        z=np.nan_to_num(z,nan=0.0,posinf=1e6,neginf=-1e6)
        if np.std(z)<1e-8: continue
        key=np.round((z-np.mean(z))/(np.std(z)+1e-9),8).tobytes()
        if key in seen: continue
        seen.add(key); evals+=1
        corr=abs(float(np.corrcoef(z,residual)[0,1])) if np.std(residual)>1e-12 else 0.0
        score=corr - 0.002*mdl_cost(e)
        scored.append((score,e))
        if score>best_score: best_score=score;best=e
        if len(scored)%50==0:
            scored=sorted(scored,key=lambda t:t[0],reverse=True)[:32]
            pool=[e for _,e in scored] + [var(v) for v in variables]

    if best is None:
        return SearchResult(False,None,evals,base.rmse,base.rmse,base_val,0.0,trig,None)
    ext=base_exprs+[best]
    fit=linear_fit(ext,train_env,train_y); vr,_=apply_fit(ext,fit,val_env,val_y)
    mdl_gain=base.dl-fit.dl-mdl_cost(best)
    accepted=(vr <= 0.92*base_val and mdl_gain>0)
    sig=np.asarray(evaluate(best,val_env),float)
    sig=((sig-np.mean(sig))/(np.std(sig)+1e-9))[:128].tolist()
    return SearchResult(accepted,best if accepted else None,evals,base.rmse,fit.rmse,vr,mdl_gain,trig,sig if accepted else None)


def random_composition_control(train_env,train_y,val_env,val_y,actions,budget,seed,max_depth=6,max_nodes=11):
    variables=tuple(sorted(train_env)); base_exprs=l0_expressions(variables)
    base=linear_fit(base_exprs,train_env,train_y); base_val,_=apply_fit(base_exprs,base,val_env,val_y)
    rng=np.random.default_rng(seed); pool=[var(v) for v in variables]; seen=set(); best_expr=None; best_val=base_val; evals=0
    while evals<budget:
        e=mutate_expr(pool,variables,rng,max_depth,max_nodes)
        try:z=np.asarray(evaluate(e,train_env),float)
        except Exception:continue
        if np.std(z)<1e-8:continue
        key=np.round((z-z.mean())/(z.std()+1e-9),8).tobytes()
        if key in seen:continue
        seen.add(key);evals+=1
        fit=linear_fit(base_exprs+[e],train_env,train_y); vr,_=apply_fit(base_exprs+[e],fit,val_env,val_y)
        if vr<best_val:best_val=vr;best_expr=e
    return best_val,best_expr,evals
