from __future__ import annotations
# BENCHMARK-ONLY module. Learner modules must never import this file.
import numpy as np
from .v419_dsl import Expr, var

FAMILIES=('circular','rank_interaction','null','noise','actuator_shuffle')

def _causal_mean(x,w):
    out=np.empty_like(x); cs=np.cumsum(np.r_[0.0,x])
    for i in range(len(x)):
        j=max(0,i-w+1);out[i]=(cs[i+1]-cs[j])/(i-j+1)
    return out

def _rank(x,w):
    out=np.empty_like(x)
    for i in range(len(x)):
        j=max(0,i-w+1);h=x[j:i+1];out[i]=(np.sum(h<x[i])+.5*np.sum(h==x[i]))/len(h)
    return out

def hidden_feature(family:str, env:dict[str,np.ndarray]) -> np.ndarray:
    if family=='circular':
        x=env['x0'];d=x-np.r_[x[0],x[:-1]]
        return d + (d < -0.5).astype(float) - (d > 0.5).astype(float)
    if family=='rank_interaction':
        x0=env['x0'];x1=env['x1']; lag=np.r_[x1[0],x1[:-1]]
        m=_causal_mean(lag,3); dm=m-np.r_[m[0],m[:-1]]
        gate=(_rank(x0,9)>.75).astype(float)
        return dm*gate
    if family=='null': return env['x0']
    raise ValueError(family)

def reference_expr(family:str) -> Expr:
    if family=='circular':
        x=var('x0'); d=Expr('diff',(x,),1)
        pos=Expr('threshold_gt',(d,),.5)
        neg=Expr('threshold_gt',(Expr('negate',(d,)),),.5)
        return Expr('add',(Expr('subtract',(d,pos)),neg))
    if family=='rank_interaction':
        x0=var('x0');x1=var('x1')
        left=Expr('diff',(Expr('filter_mean',(Expr('lag',(x1,),1),),3),),1)
        gate=Expr('threshold_gt',(Expr('rank01',(x0,),9),),.75)
        return Expr('interaction',(left,gate))
    raise ValueError(family)

def make_world(family:str,seed:int,n:int=700,variant:int=0):
    r=np.random.default_rng(seed); t=np.arange(n); u=r.normal(size=(n,2))
    if family=='noise':
        env={f'x{i}':r.normal(size=n) for i in range(3)}; y=r.normal(scale=.8,size=n);return env,u,y
    if family in ('circular','actuator_shuffle'):
        step=.035+.018*np.tanh(u[:,0])+.01*r.normal(size=n)
        phase=np.cumsum(step); x0=np.mod(phase,1.0)
        env={'x0':x0,'x1':r.normal(size=n),'x2':.6*np.sin(.07*t)+.25*r.normal(size=n)}
        h=hidden_feature('circular',env)
        action=u[:,0].copy()
        if family=='actuator_shuffle': action=r.permutation(action)
        y=1.5*h+.22*action+.08*r.normal(size=n)
        if variant:y += .12*np.sin(.013*t)
        return env,u,y
    if family=='rank_interaction':
        raw=np.zeros(n)
        for i in range(1,n):raw[i]=.88*raw[i-1]+.35*r.normal()
        x0=np.tanh(raw) if variant==0 else np.sign(raw)*np.abs(raw)**.55
        x1=np.cumsum(.08*r.normal(size=n)+.025*u[:,1])
        env={'x0':x0,'x1':x1,'x2':r.normal(size=n)}
        h=hidden_feature(family,env);y=1.6*h+.18*u[:,1]+.06*r.normal(size=n)
        return env,u,y
    if family=='null':
        x0=np.zeros(n)
        for i in range(1,n):x0[i]=.8*x0[i-1]+.25*r.normal()
        x1=r.normal(size=n);x2=r.normal(size=n)
        env={'x0':x0,'x1':x1,'x2':x2}
        lag=np.r_[x0[0],x0[:-1]];diff=x0-lag;mean=_causal_mean(x1,3)
        y=.7*lag+.5*diff+.25*mean+.15*(x0*x2)+.08*u[:,0]+.08*r.normal(size=n)
        return env,u,y
    raise ValueError(family)
