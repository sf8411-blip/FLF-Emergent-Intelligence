from __future__ import annotations
import math
import numpy as np
from . import v420_repair as _p
from . import v420_repair_iter1 as _i1
from .v420_repair import *

# Scientific repair iteration 2. Fixed BEFORE any iteration-2 result.
# Criteria/H1/H2 are unchanged.
SEARCH_BUDGET=1024
_i1_make_world=_i1.make_world
_SPURIOUS_OFFSETS={'x0':101,'x1':113,'x2':127,'d0':139,'d1':151,'d2':163}

def _ar(r,n,rho,sd=1.0):
    x=np.zeros(n);innov=sd*math.sqrt(max(1-rho*rho,1e-6));x[0]=r.normal(scale=sd)
    for i in range(1,n):x[i]=rho*x[i-1]+r.normal(scale=innov)
    return x

def make_world(family,seed,n=700,action_scale=1.0,meta=None):
    seed=int(seed)
    if family in ('corrupt_circular','corrupt_rank'):
        bf='circular' if family=='corrupt_circular' else 'rank'
        env,y,h,m=make_world(bf,seed,n=n,action_scale=action_scale,meta=meta)
        rr=np.random.default_rng(seed+818181);shift=int(rr.integers(len(y)//4,3*len(y)//4))
        return env,np.roll(y,shift),np.zeros_like(h),{**m,'corruption':'circular_shift_y','shift':shift}
    if family=='rank':
        r=np.random.default_rng(seed);u=r.normal(size=(n,2))*float(action_scale);m=dict(meta or {})
        wf=float(m.get('filter_tau',r.uniform(2.2,5.8)));wr=float(m.get('rank_tau',r.uniform(6.2,14.8)));q=float(m.get('quantile',r.uniform(.48,.62)))
        if min(abs(wf-z) for z in (2.,3.,5.,9.))<.03:wf+=.071
        if min(abs(wr-z) for z in (2.,3.,5.,9.))<.03:wr+=.071
        if min(abs(q-z) for z in (.25,.5,.75))<.015:q+=.031
        m={'filter_tau':wf,'rank_tau':wr,'quantile':q};raw=np.zeros(n)
        for i in range(1,n):raw[i]=.88*raw[i-1]+.35*r.normal()
        x0=np.tanh(raw);x1=np.cumsum(.08*r.normal(size=n)+.025*u[:,1]);env={'x0':x0,'x1':x1,'x2':r.normal(size=n),'u0':u[:,0],'u1':u[:,1]}
        h=_p.hidden_feature('rank',env,m);y=8.0*h+.18*u[:,1]+.04*r.normal(size=n)
        return env,y,h,m
    if family=='spurious':
        n=max(n,1200);rho=float((meta or {}).get('rho',.985));r=np.random.default_rng(seed);u=r.normal(size=(n,2))*float(action_scale);env={'u0':u[:,0],'u1':u[:,1]}
        for k,off in _SPURIOUS_OFFSETS.items():env[k]=_ar(np.random.default_rng(seed+off),n,rho,1.0)
        y=_ar(np.random.default_rng(seed+999),n,rho,1.0)
        return env,y,np.zeros(n),{'rho':rho,'deterministic_channel_offsets':dict(_SPURIOUS_OFFSETS)}
    return _i1_make_world(family,seed,n=n,action_scale=action_scale,meta=meta)

# The inherited protocol/search functions resolve make_world in the base module
# at runtime; patch only the live DEV module, leaving iteration-0 source bytes intact.
_p.make_world=make_world

def run_seed_core(family,seed,challenge_B=999,search_budget=None):
    return _p.run_seed_core(family,seed,challenge_B=challenge_B,search_budget=SEARCH_BUDGET if search_budget is None else int(search_budget))

def select_guided(env,y,base0,seed,count=None,shuffle=False):
    return _p.select_guided(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count),shuffle)

def select_controls(env,y,base0,seed,count=None):
    return _p.select_controls(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count))
