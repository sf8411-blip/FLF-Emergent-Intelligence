from __future__ import annotations
import numpy as np
from . import v420_repair as _p
from . import v420_dev as base
from .v420_repair import *

# Repair iteration 1. Criteria are unchanged. These are instrument/world-power
# repairs fixed after archiving iteration-0 failures.
SEARCH_BUDGET=512
_prev_make_world=_p.make_world

def make_world(family,seed,n=700,action_scale=1.0,meta=None):
    seed=int(seed)
    # R-D: increase observation length rather than relaxing the 1.10*noise-SD gate.
    if family=='null' and n==700:
        return _prev_make_world(family,seed,n=1200,action_scale=action_scale,meta=meta)
    # R4 Rank: iteration 0 showed near-degenerate high-quantile gates. Move the
    # WORLD away from almost-closed gates; grammar and gate criteria are unchanged.
    if family=='rank':
        r=np.random.default_rng(seed);u=r.normal(size=(n,2))*float(action_scale);m=dict(meta or {})
        wf=float(m.get('filter_tau',r.uniform(2.2,5.8)));wr=float(m.get('rank_tau',r.uniform(6.2,14.8)));q=float(m.get('quantile',r.uniform(.50,.70)))
        if min(abs(wf-z) for z in (2.,3.,5.,9.))<.03:wf+=.071
        if min(abs(wr-z) for z in (2.,3.,5.,9.))<.03:wr+=.071
        if min(abs(q-z) for z in (.25,.5,.75))<.015:q+=.031
        m={'filter_tau':wf,'rank_tau':wr,'quantile':q};raw=np.zeros(n)
        for i in range(1,n):raw[i]=.88*raw[i-1]+.35*r.normal()
        x0=np.tanh(raw);x1=np.cumsum(.08*r.normal(size=n)+.025*u[:,1]);env={'x0':x0,'x1':x1,'x2':r.normal(size=n),'u0':u[:,0],'u1':u[:,1]};h=_p.hidden_feature('rank',env,m);y=5.5*h+.18*u[:,1]+.04*r.normal(size=n);return env,y,h,m
    # corrupt-Rank calls the patched Rank world through the old corruption function.
    return _prev_make_world(family,seed,n=n,action_scale=action_scale,meta=meta)

# Patch the previous module's global dispatch so inherited functions (challenge,
# run_seed_core) call the iteration-1 world definition. This is process-local and
# deterministic; iteration-0 source remains untouched for auditability.
_p.make_world=make_world


def run_seed_core(family,seed,challenge_B=999,search_budget=None):
    return _p.run_seed_core(family,seed,challenge_B=challenge_B,search_budget=SEARCH_BUDGET if search_budget is None else int(search_budget))

def select_guided(env,y,base0,seed,count=None,shuffle=False):
    return _p.select_guided(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count),shuffle)

def select_controls(env,y,base0,seed,count=None):
    return _p.select_controls(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count))
