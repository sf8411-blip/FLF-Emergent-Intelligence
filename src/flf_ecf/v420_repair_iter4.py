from __future__ import annotations
import math
import numpy as np
from . import v420_repair as _p
from .v420_repair import *

# FINAL allowed scientific repair iteration 4. Fixed BEFORE any iteration-4 result.
# Criteria/H1/H2 thresholds remain unchanged. No further repair iteration exists.
SEARCH_BUDGET=1024
_BASE_MAKE_WORLD=_p.make_world
_SPURIOUS_OFFSETS={
    'x0':101,'x1':113,'x2':127,'d0':139,'d1':151,'d2':163,
    'd3':173,'d4':181,'d5':193,'d6':211,'d7':223,'d8':233,
    'd9':239,'d10':251,'d11':263,'d12':277,'d13':281,'d14':293,'d15':307,
}

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
        rng=np.random.default_rng(seed);u=rng.normal(size=(n,2))*float(action_scale);m=dict(meta or {})
        wf=float(m.get('filter_tau',rng.uniform(2.2,5.8)))
        wr=float(m.get('rank_tau',rng.uniform(6.2,14.8)))
        q=float(m.get('quantile',rng.uniform(.48,.62)))
        if min(abs(wf-z) for z in (2.,3.,5.,9.))<.03:wf+=.071
        if min(abs(wr-z) for z in (2.,3.,5.,9.))<.03:wr+=.071
        if min(abs(q-z) for z in (.25,.5,.75))<.015:q+=.031
        m={'filter_tau':wf,'rank_tau':wr,'quantile':q}
        raw=np.zeros(n)
        for i in range(1,n):raw[i]=.88*raw[i-1]+.35*rng.normal()
        x0=np.tanh(raw);x1=np.cumsum(.08*rng.normal(size=n)+.025*u[:,1])
        env={'x0':x0,'x1':x1,'x2':rng.normal(size=n),'u0':u[:,0],'u1':u[:,1]}
        h=_p.hidden_feature('rank',env,m)
        # Only observability is strengthened relative to iteration 2; hidden
        # structure/grammar stay unchanged.
        y=20.0*h+.18*u[:,1]+.04*rng.normal(size=n)
        return env,y,h,m
    if family=='null':
        return _BASE_MAKE_WORLD('null',seed,n=max(n,1200),action_scale=action_scale,meta=meta)
    if family=='spurious':
        # Intentionally hard false-positive world: highly persistent independent
        # traces and many persistent distractors. Fresh traces remain independent.
        n=max(n,600);rho=float((meta or {}).get('rho',.998));rng=np.random.default_rng(seed)
        u=rng.normal(size=(n,2))*float(action_scale);env={'u0':u[:,0],'u1':u[:,1]}
        for k,off in _SPURIOUS_OFFSETS.items():env[k]=_ar(np.random.default_rng(seed+off),n,rho,1.0)
        y=_ar(np.random.default_rng(seed+999),n,rho,1.0)
        return env,y,np.zeros(n),{'rho':rho,'deterministic_channel_offsets':dict(_SPURIOUS_OFFSETS)}
    return _BASE_MAKE_WORLD(family,seed,n=n,action_scale=action_scale,meta=meta)

def challenge(e,family,meta,seed,B=999):
    env,y,h,m=make_world(family,int(seed),action_scale=1.6,meta=meta)
    b=_p.b0_and_oracle(env,y,h);z=_p.evaluate(e,env);fit=_p.extension_fit(b,y,z)
    actual=b['b0_val_rmse']-fit['val_rmse'];tr=b['tr']
    acl=max(_p.acf_length(z[tr]),_p.acf_length(b['residual'][tr]))
    # Conservative block choice: never shorter than measured ACF length.
    block=min(max(20,int(math.ceil(1.25*acl))),max(20,len(tr)-1))
    if block<acl: block=int(acl)
    rng=np.random.default_rng(int(seed)+404);null=[]
    for _ in range(int(B)):
        zp=_p.base.block_permute(z,rng,block)
        null.append(b['b0_val_rmse']-_p.extension_fit(b,y,zp)['val_rmse'])
    pv=(1+sum(x>=actual for x in null))/(B+1)
    return float(pv),float(actual),{'acf_length':int(acl),'block_length':int(block),'meta':m}

# Patch only the live DEV protocol dispatch. Historical iteration source/results
# are immutable and remain archived.
_p.make_world=make_world
_p.challenge=challenge

def run_seed_core(family,seed,challenge_B=999,search_budget=None):
    return _p.run_seed_core(family,seed,challenge_B=challenge_B,search_budget=SEARCH_BUDGET if search_budget is None else int(search_budget))

def select_guided(env,y,base0,seed,count=None,shuffle=False):
    return _p.select_guided(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count),shuffle)

def select_controls(env,y,base0,seed,count=None):
    return _p.select_controls(env,y,base0,seed,SEARCH_BUDGET if count is None else int(count))
