from __future__ import annotations
from dataclasses import dataclass
import math, random
import numpy as np
from .v418 import G, Model, rg, mut, cross
from .v419_dsl import Expr, evaluate, l0_expressions

@dataclass
class AdapterFit:
    exprs: list[Expr]
    genome: G
    model: Model
    train_residual: np.ndarray
    train_rmse: float
    validation_rmse: float
    test_rmse: float
    description_length: float
    model_evaluations: int


def _safe_series(e, env):
    z=np.asarray(evaluate(e,env),float)
    return np.nan_to_num(z,nan=0.0,posinf=1e6,neginf=-1e6)


def select_three(pool:list[Expr], env:dict[str,np.ndarray], y:np.ndarray) -> list[Expr]:
    scored=[]
    for e in pool:
        try:z=_safe_series(e,env)
        except Exception:continue
        score=abs(float(np.corrcoef(z,y)[0,1])) if np.std(z)>1e-10 and np.std(y)>1e-10 else 0.0
        scored.append((score,e.text(),e))
    scored.sort(key=lambda t:(t[0],t[1]),reverse=True)
    chosen=[e for _,_,e in scored[:3]]
    if not chosen:raise ValueError('empty feature pool')
    while len(chosen)<3:chosen.append(chosen[-1])
    return chosen


def _rows(exprs, env, u, y):
    F=np.column_stack([_safe_series(e,env) for e in exprs]);Y=np.repeat(np.asarray(y,float)[:,None],3,axis=1)
    return [(F[i],np.asarray(u[i],float),Y[i],0.0) for i in range(len(y))]


def _scalar_predictions(model:Model, rows):
    xs=[z[0] for z in rows];pred=[];truth=[]
    for i,(x,u,y,r) in enumerate(rows):
        h=xs[max(0,i-model.g.w+1):i+1];pred.append(float(np.mean(model.pred(h,u))));truth.append(float(np.mean(y)))
    return np.asarray(pred),np.asarray(truth)


def _rmse(model, rows):
    p,y=_scalar_predictions(model,rows);return float(np.sqrt(np.mean((p-y)**2)))


def _sanitize(g:G) -> G:
    # v4.18 target=1 predicts a delta in observation coordinates. The adapter target
    # is a separate scalar repeated in three coordinates, so target=0 is the valid mode.
    # policy is irrelevant to the prediction-only downstream metric and is fixed to 0.
    return G(g.q,g.w,g.f,g.ks,g.ka,g.norm,0,0,g.actmode,g.mdl)


def _complexity(model:Model) -> float:
    g=model.g
    return (g.ks*g.ka + model.n/max(model.n,1))/100 + (0,.006,.015,.03)[g.mdl]*(g.ks+g.ka+g.w+g.actmode)


def fit_v418_l0(train_env,train_u,train_y,val_env,val_u,val_y,test_env,test_u,test_y,
                 extra_exprs:list[Expr]|None=None,seed:int=0,evaluation_budget:int=48) -> AdapterFit:
    """Use unchanged v4.18.5 Model/G/mutation/crossover over an L0-derived feature pool.

    `extra_exprs` is empty for B0, contains the benchmark-only oracle expression for B1,
    and contains only the residual-synthesized expression for L1/B2/B3. Feature selection
    uses training data only.
    """
    pool=l0_expressions(sorted(train_env)) + list(extra_exprs or [])
    exprs=select_three(pool,train_env,train_y)
    tr=_rows(exprs,train_env,train_u,train_y);va=_rows(exprs,val_env,val_u,val_y);te=_rows(exprs,test_env,test_u,test_y)
    rr=random.Random(seed);seen=set();pop=[];evaluated={}
    def score(g):
        g=_sanitize(g)
        if g not in evaluated:
            m=Model(g).fit(tr);evaluated[g]=(_rmse(m,va)+.07*_complexity(m),m)
        return evaluated[g][0]
    while len(pop)<10 and len(seen)<evaluation_budget:
        g=_sanitize(rg(rr))
        if g not in seen:seen.add(g);pop.append(g);score(g)
    while len(seen)<evaluation_budget:
        elite=sorted(pop,key=score)[:4];new=elite[:];tries=0
        while len(new)<10 and len(seen)<evaluation_budget and tries<200:
            child=_sanitize(cross(rr.choice(elite),rr.choice(elite),rr));child=_sanitize(mut(child,rr));tries+=1
            if child not in seen:seen.add(child);new.append(child);score(child)
        if len(new)==len(elite):
            while len(seen)<evaluation_budget:
                g=_sanitize(rg(rr))
                if g not in seen:seen.add(g);new.append(g);score(g)
                if len(new)>=10:break
        pop=new[:10]
    best=min(evaluated,key=lambda g:evaluated[g][0]);model=evaluated[best][1]
    p,y=_scalar_predictions(model,tr);res=y-p
    trrm=float(np.sqrt(np.mean(res*res)));vr=_rmse(model,va);ter=_rmse(model,te)
    var=max(float(np.var(res)),1e-12)
    dl=.5*len(res)*math.log1p(var/1e-4)+.5*(best.ks*best.ka+len(exprs))*math.log(max(len(res),2))
    return AdapterFit(exprs,best,model,res,trrm,vr,ter,dl,len(evaluated))
