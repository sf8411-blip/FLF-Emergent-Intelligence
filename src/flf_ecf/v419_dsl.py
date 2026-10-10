from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable
import numpy as np

# Learner-facing low-level DSL only. No benchmark/world imports are permitted here.

@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple['Expr', ...] = ()
    param: float | int | str | None = None

    def depth(self) -> int:
        return 0 if not self.args else 1 + max(a.depth() for a in self.args)

    def nodes(self) -> int:
        return 1 + sum(a.nodes() for a in self.args)

    def primitives(self) -> frozenset[str]:
        s = {self.op}
        for a in self.args:
            s |= set(a.primitives())
        return frozenset(s)

    def text(self) -> str:
        if self.op == 'var':
            return str(self.param)
        p = '' if self.param is None else f'[{self.param}]'
        return f"{self.op}{p}(" + ','.join(a.text() for a in self.args) + ')'


def var(name: str) -> Expr:
    return Expr('var', (), name)


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0: return x.copy()
    out = np.empty_like(x, dtype=float)
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


def _diff(x: np.ndarray, k: int) -> np.ndarray:
    return x.astype(float) - _lag(x.astype(float), k)


def _filter(x: np.ndarray, w: int) -> np.ndarray:
    x=np.asarray(x,float); out=np.empty_like(x)
    cs=np.cumsum(np.r_[0.0,x])
    for i in range(len(x)):
        j=max(0,i-w+1); out[i]=(cs[i+1]-cs[j])/(i-j+1)
    return out


def _rank01(x: np.ndarray, w: int) -> np.ndarray:
    x=np.asarray(x,float); out=np.empty_like(x)
    for i in range(len(x)):
        j=max(0,i-w+1); h=x[j:i+1]
        out[i]=(np.sum(h < x[i]) + 0.5*np.sum(h == x[i]))/len(h)
    return out


def evaluate(expr: Expr, env: dict[str,np.ndarray]) -> np.ndarray:
    if expr.op == 'var':
        return np.asarray(env[str(expr.param)], float)
    xs=[evaluate(a,env) for a in expr.args]
    op=expr.op; p=expr.param
    if op=='lag': return _lag(xs[0], int(p))
    if op=='diff': return _diff(xs[0], int(p))
    if op=='cumsum': return np.cumsum(xs[0])
    if op=='filter_mean': return _filter(xs[0], int(p))
    if op=='rank01': return _rank01(xs[0], int(p))
    if op=='threshold_gt': return (xs[0] > float(p)).astype(float)
    if op=='threshold_lt': return (xs[0] < float(p)).astype(float)
    if op=='modulo': return np.mod(xs[0], float(p))
    if op=='product': return xs[0]*xs[1]
    if op=='interaction': return xs[0]*(xs[1] > 0.5)
    if op=='negate': return -xs[0]
    if op=='abs': return np.abs(xs[0])
    if op=='add': return xs[0]+xs[1]
    if op=='subtract': return xs[0]-xs[1]
    raise ValueError(op)

UNARY = (
    ('lag',(1,2,3)), ('diff',(1,2)), ('cumsum',(None,)),
    ('filter_mean',(2,3,5)), ('rank01',(5,9)),
    ('threshold_gt',(0.25,0.5,0.75)), ('threshold_lt',(0.25,0.5,0.75)),
    ('modulo',(1.0,2.0,6.283185307179586)), ('negate',(None,)), ('abs',(None,)),
)
BINARY=('product','interaction','add','subtract')


def mutate_expr(pool: list[Expr], variables: tuple[str,...], rng: np.random.Generator, max_depth:int=6, max_nodes:int=11) -> Expr:
    """Generate one typed-enough numeric expression. Search has no benchmark-specific macro."""
    for _ in range(100):
        mode=int(rng.integers(3))
        if mode==0 or not pool:
            e=var(variables[int(rng.integers(len(variables)))])
            steps=int(rng.integers(1,max_depth+1))
            for _ in range(steps):
                op, params = UNARY[int(rng.integers(len(UNARY)))]
                p=params[int(rng.integers(len(params)))]
                e=Expr(op,(e,),p)
                if e.depth()>=max_depth: break
        elif mode==1:
            a=pool[int(rng.integers(len(pool)))]; op,params=UNARY[int(rng.integers(len(UNARY)))]
            p=params[int(rng.integers(len(params)))]; e=Expr(op,(a,),p)
        else:
            a=pool[int(rng.integers(len(pool)))]; b=pool[int(rng.integers(len(pool)))]; op=BINARY[int(rng.integers(len(BINARY)))]
            e=Expr(op,(a,b),None)
        if e.depth()<=max_depth and e.nodes()<=max_nodes:
            return e
    return var(variables[0])


def mdl_cost(expr: Expr) -> float:
    learned_constants = 0
    return expr.nodes() + 0.25*len(expr.primitives()) + 0.5*learned_constants


def l0_expressions(variables: Iterable[str]) -> list[Expr]:
    out=[]
    for name in variables:
        x=var(name); out += [x, Expr('lag',(x,),1), Expr('diff',(x,),1)]
        out += [Expr('filter_mean',(x,),w) for w in (2,3,5)]
    names=list(variables)
    for i,a in enumerate(names):
        for b in names[i+1:]: out.append(Expr('product',(var(a),var(b))))
    return out
