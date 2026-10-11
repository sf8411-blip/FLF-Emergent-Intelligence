from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .v419_dsl import Expr, evaluate, l0_expressions, mutate_expr, mdl_cost, var

# IMPORTANT: this module must not import benchmark generators, oracle code, or v4.18.

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
    rmse=float(np.sqrt(np.mean(res*res)));var=max(float(np.var(res)),1e-12)
    dl=0.5*len(y)*math.log1p(var/1e-4)+0.5*X.shape[1]*math.log(max(len(y),2))
    return Fit(coef,pred,res,rmse,dl)


def apply_fit(exprs:list[Expr], fit:Fit, env:dict[str,np.ndarray], target:np.ndarray) -> tuple[float,np.ndarray]:
    X=_matrix(exprs,env); pred=X@fit.coef; res=np.asarray(target)-pred
    return float(np.sqrt(np.mean(res*res))),res


def _acf1(x:np.ndarray)->float:
    x=np.asarray(x,float)-np.mean(x)
    if len(x)<4 or np.std(x)<1e-12:return 0.0
    z=float(np.corrcoef(x[:-1],x[1:])[0,1])
    return z if np.isfinite(z) else 0.0


def _fisher_abs_z(a:np.ndarray,b:np.ndarray)->float:
    """Approximate absolute z score for linear dependence under rho=0."""
    a=np.asarray(a,float);b=np.asarray(b,float);n=min(len(a),len(b))
    if n<5 or np.std(a[:n])<1e-12 or np.std(b[:n])<1e-12:return 0.0
    rho=float(np.corrcoef(a[:n],b[:n])[0,1])
    if not np.isfinite(rho):return 0.0
    rho=float(np.clip(rho,-.999999,.999999))
    return abs(float(np.arctanh(rho)))*math.sqrt(max(n-3,1))


def _compression_gap_z(residual:np.ndarray)->float:
    """Likelihood-ratio z-like score for compressible sign-transition structure.

    This uses only the learner's own residual sequence.  Under an iid-sign null,
    the 2x2 transition likelihood-ratio statistic is asymptotically chi-square(1);
    sqrt(G^2) is therefore an absolute z-like compression-gap statistic.  It is
    distinct from the linear lag-1 autocorrelation statistic and requires no
    benchmark/world metadata.
    """
    r=np.asarray(residual,float)
    if len(r)<8 or np.std(r)<1e-12:return 0.0
    bits=(r>=np.median(r)).astype(int)
    table=np.zeros((2,2),float)
    for a,b in zip(bits[:-1],bits[1:]):table[a,b]+=1.0
    n=float(table.sum())
    if n<=0:return 0.0
    rs=table.sum(1);cs=table.sum(0);g2=0.0
    for i in range(2):
        for j in range(2):
            exp=rs[i]*cs[j]/n
            obs=table[i,j]
            if obs>0 and exp>0:g2+=2.0*obs*math.log(obs/exp)
    return math.sqrt(max(g2,0.0))


def trigger_components(residual:np.ndarray, actions:np.ndarray, base_dl:float=0.0) -> tuple[float,float,float,float]:
    """Preregistered residual trigger.

    score = z(residual_autocorrelation)
          + z(intervention_effect_inconsistency)
          + z(compression_gap)

    All three components are computed exclusively from learner-visible residuals
    and actions. ``base_dl`` is retained for API compatibility; no oracle or
    generator quantity is used.
    """
    r=np.asarray(residual,float);n=len(r)
    ac=_fisher_abs_z(r[:-1],r[1:]) if n>=5 else 0.0
    inc=max((_fisher_abs_z(np.asarray(actions)[:,j],r) for j in range(actions.shape[1])),default=0.0)
    comp=_compression_gap_z(r)
    score=ac+inc+comp
    return score,ac,inc,comp

@dataclass
class Proposal:
    expr: Expr|None
    evaluations: int
    trigger_score: float
    candidate_score: float


def propose_operator(train_env:dict[str,np.ndarray], residual:np.ndarray, actions:np.ndarray,
                     budget:int, seed:int, shuffled_residual:bool=False, trigger_enabled:bool=True,
                     max_depth:int=6, max_nodes:int=11) -> Proposal:
    """Residual-guided composition search over the generic primitive DSL."""
    variables=tuple(sorted(train_env)); r=np.asarray(residual,float).copy(); rng=np.random.default_rng(seed)
    if shuffled_residual:rng.shuffle(r)
    trig,*_=trigger_components(r,actions,0.0)
    if (not trigger_enabled) or trig<2.5:return Proposal(None,0,trig,float('-inf'))
    pool=[var(v) for v in variables];scored=[];seen=set();evals=0;best=None;best_score=-1e99
    while evals<budget:
        e=mutate_expr(pool,variables,rng,max_depth,max_nodes)
        try:z=np.asarray(evaluate(e,train_env),float)
        except Exception:continue
        z=np.nan_to_num(z,nan=0.0,posinf=1e6,neginf=-1e6)
        if np.std(z)<1e-8:continue
        zn=(z-z.mean())/(z.std()+1e-9);key=np.round(zn,8).tobytes()
        if key in seen:continue
        seen.add(key);evals+=1
        corr=abs(float(np.corrcoef(zn,r)[0,1])) if np.std(r)>1e-12 else 0.0
        score=corr-.002*mdl_cost(e);scored.append((score,e))
        if score>best_score:best_score=score;best=e
        if len(scored)%50==0:
            scored=sorted(scored,key=lambda t:t[0],reverse=True)[:32]
            pool=[e for _,e in scored]+[var(v) for v in variables]
    return Proposal(best,evals,trig,best_score)


def random_proposal(train_env:dict[str,np.ndarray], residual:np.ndarray, budget:int, seed:int,
                    max_depth:int=6,max_nodes:int=11) -> Proposal:
    """Equal candidate-evaluation random-composition control.

    Candidate generation remains random; the same learner-visible residual statistic is
    used only to select the best of the uniformly generated candidates for the one
    downstream v4.18.5 evaluation.
    """
    variables=tuple(sorted(train_env));r=np.asarray(residual,float);rng=np.random.default_rng(seed)
    pool=[var(v) for v in variables];seen=set();evals=0;best=None;best_score=-1.0
    while evals<budget:
        e=mutate_expr(pool,variables,rng,max_depth,max_nodes)
        try:z=np.asarray(evaluate(e,train_env),float)
        except Exception:continue
        z=np.nan_to_num(z,nan=0.0,posinf=1e6,neginf=-1e6)
        if np.std(z)<1e-8:continue
        zn=(z-z.mean())/(z.std()+1e-9);key=np.round(zn,8).tobytes()
        if key in seen:continue
        seen.add(key);evals+=1
        score=abs(float(np.corrcoef(zn,r)[0,1])) if np.std(r)>1e-12 else 0.0
        if score>best_score:best_score=score;best=e
    return Proposal(best,evals,float('nan'),best_score)
