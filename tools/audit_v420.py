#!/usr/bin/env python3
from __future__ import annotations
import json, math, pathlib, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_dev as v
from flf_ecf import v420_dress as d
from flf_ecf import v420_protocol as p

DRESS_RANGES=[range(320000,320032),range(321000,321032),range(322000,322032),range(323000,323032),range(324000,324032),range(325000,325032),range(326000,326032)]
def assert_non_dress(seed:int):
    assert not any(seed in r for r in DRESS_RANGES),seed

def acf1(x):
    x=np.asarray(x,float);return abs(v.corr(x[1:],x[:-1]))

def pooled(rows,key):
    den=sum(r['den'] for r in rows);return sum(r[key] for r in rows)/max(den,1e-12)

def boot_delta(rows,ka,kb,B=1500,seed=991):
    rr=np.random.default_rng(seed);n=len(rows);vals=[]
    for _ in range(B):
        ix=rr.integers(0,n,n);s=[rows[int(i)] for i in ix]
        vals.append(pooled(s,ka)-pooled(s,kb))
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def trigger_probe(family,seed):
    assert_non_dress(seed)
    env,y,h,meta=d.make_dress_world(family,seed)
    base=d._gated_base(env,y,h)
    actions=np.column_stack([env['u0'],env['u1']])
    obs,thr,opened=v.calibrated_trigger(base['residual'][base['tr']],actions[base['tr']],seed+101,B=999)
    return bool(opened),acf1(h[base['tr']]) if np.std(h[base['tr']])>1e-12 else 0.0

def program_text_probe():
    env,_,_,_=v.make_world('circular',410001)
    progs=[v.Program('moddiff','x0',p=1.1),v.Program('mean','x0',p=3.0),v.Program('prod','x0',b='x1'),v.Program('diff','x0'),v.Program('modraw','x0',p=1.1),v.Program('rankprod','x0',b='x1',p=9.0,q=.7)]
    out={}
    for q in progs:
        try:q.text();out[q.kind]='OK'
        except Exception as e:out[q.kind]=type(e).__name__
    return out

def c1():
    fams=('circular','rank','null','corrupt_circular','corrupt_rank','noise')
    ans={};ac=[]
    for j,f in enumerate(fams):
        opens=0
        for i in range(40):
            seed=411000+j*1000+i;op,a=trigger_probe(f,seed);opens+=op
            if f in ('circular','rank'):ac.append(a)
        ans[f]={'opens':int(opens),'n':40}
    return {'families':ans,'median_abs_hidden_lag1_acf':float(np.median(ac))}

def c2():
    txt=program_text_probe();cr=0;total=24;errs={}
    fams=('circular','rank','null','corrupt_circular','corrupt_rank','noise')
    for j in range(total):
        f=fams[j%len(fams)];idx=1000+j # run_seed adds dress base -> prohibited; do not execute run_seed directly.
        # Reproduce the crash path on a non-dress world by calling the same selected-arm serializer.
        seed=412000+j;assert_non_dress(seed)
        env,y,h,meta=d.make_dress_world('circular' if f=='noise' else f,seed)
        base=d._gated_base(env,y,h)
        try:
            sel=d._select_guided(env,y,base,seed+1001,shuffle=False)
            d._eval_arm(sel,base,y,'circular' if f=='noise' else f,meta,seed+700001)
        except Exception as e:
            cr+=1;errs[type(e).__name__]=errs.get(type(e).__name__,0)+1
    return {'program_text':txt,'crash_path':{'crashed':cr,'n':total,'errors':errs},'note':'run_seed itself is not called because it hardcodes prohibited dress seed bases; identical serializer path is exercised on non-dress worlds.'}

def c3():
    gaps={};null_rm=[]
    for f,base_seed in [('circular',413000),('rank',414000),('null',415000)]:
        vals=[]
        for i in range(40):
            seed=base_seed+i;assert_non_dress(seed)
            env,y,h,_=v.make_world(f,seed);X,_=v.l0_matrix(env);tr,val,te=v.splits(len(y));m=v.fit_complete(X,y,tr,val);c=v.fit_linear(X,y,tr,1e-8);ce=v.rmse(y[te],v.predict(c,X)[te]);vals.append((v.rmse(y[te],v.predict(m,X)[te])-ce)/max(ce,1e-12))
            if f=='null':null_rm.append(v.rmse(y[te],v.predict(m,X)[te]))
        gaps[f]={'within_3pct_rate':float(np.mean(np.asarray(vals)<=.03)),'median_relative_gap':float(np.median(vals))}
    return {'r1_same_feature_ceiling':gaps,'null_b0_test_rmse_median':float(np.median(null_rm)),'generator_noise_sd':.08,'rmse_over_noise_sd':float(np.median(null_rm)/.08)}

def c4():
    exact={}
    for f,seed in [('circular',416001),('rank',416002)]:
        env,y,h,meta=v.make_world(f,seed);q=p.oracle_program(f,meta);exact[f]=float(np.max(np.abs(v.eval_program(q,env)-h)))
    accepted=0;right_template=0;both=0;period_err=[];n=36
    for i in range(n):
        f='circular' if i<n//2 else 'rank';seed=416100+i;assert_non_dress(seed)
        env,y,h,meta=v.make_world(f,seed);base=d._gated_base(env,y,h);sel=d._select_guided(env,y,base,seed+1001,shuffle=False)
        _,prog,z,fit,_=sel
        ok,diag=d.admit(prog,z,fit,base,y,f,meta,seed+700001)
        rt=(f=='circular' and prog.kind=='moddiff') or (f=='rank' and prog.kind=='rankprod')
        accepted+=int(ok);right_template+=int(rt);both+=int(ok and rt)
        if f=='circular' and prog.kind=='moddiff':period_err.append(abs(float(prog.p)-meta['period'])/meta['period'])
    return {'oracle_feature_max_abs_diff':exact,'sample_program_kinds':6,'candidate_budget':25,'bypass_n':n,'accepted':accepted,'right_hidden_template_drawn':right_template,'accepted_and_right_template':both,'circular_period_median_relative_error':float(np.median(period_err)) if period_err else None}

def admission_stage(prog,z,fit,base,y):
    r2=v.novelty_r2(z,base['X'],base['tr']);gain=(base['b0_val_rmse']-fit['val_rmse'])/max(base['b0_val_rmse'],1e-12)
    if gain<v.VALIDATION_GAIN_MIN:return 'gain'
    if r2>v.NOVELTY_R2_MAX:return 'novelty'
    return 'challenge'

def c5():
    counts={'gain':0,'novelty':0,'challenge':0};decisions=0
    fams=('null','corrupt_circular','corrupt_rank','noise')
    for j in range(64):
        f=fams[j%4];seed=417000+j;assert_non_dress(seed)
        env,y,h,meta=d.make_dress_world(f,seed);base=d._gated_base(env,y,h)
        g=d._select_guided(env,y,base,seed+1001,False);b2a,b2b,_=d._select_random_controls(env,y,base,seed+2001);b3=d._select_guided(env,y,base,seed+3001,True)
        arms=[g,b2a,b2b,b3]
        for sel in arms:
            if len(sel)==5:_,prog,z,fit,_=sel
            else:prog,z,fit,_=sel
            counts[admission_stage(prog,z,fit,base,y)]+=1;decisions+=1
    return {'decisions':decisions,'stages':counts}

def bypass_row(f,seed):
    env,y,h,meta=v.make_world(f,seed);base=d._gated_base(env,y,h)
    den=base['b0_test_rmse']-base['b1_test_rmse']
    g=d._select_guided(env,y,base,seed+1001,False);b2a,b2b,_=d._select_random_controls(env,y,base,seed+2001)
    def val(sel):
        if len(sel)==5:_,prog,z,fit,_=sel
        else:prog,z,fit,_=sel
        ok,_=d.admit(prog,z,fit,base,y,f,meta,seed+700001)
        return base['b0_test_rmse']-(fit['test_rmse'] if ok else base['b0_test_rmse']),evsig(prog)
    gn,gs=val(g);an,asig=val(b2a);bn,bsig=val(b2b)
    return {'den':den,'g':gn,'a':an,'b':bn,'same':asig==bsig}
def evsig(q):return (q.kind,q.a,q.b,q.p,q.q)

def c6():
    rows=[]
    for i in range(96):
        f='circular' if i<48 else 'rank';seed=418000+i;assert_non_dress(seed);rows.append(bypass_row(f,seed))
    whole={'guided':pooled(rows,'g'),'b2a':pooled(rows,'a'),'b2b':pooled(rows,'b'),'same_program':sum(r['same'] for r in rows)}
    rr=np.random.default_rng(419999);h1=0;h2a=0;h2b=0;reps=300
    for _ in range(reps):
        ids=np.r_[rr.choice(48,32,replace=False),48+rr.choice(48,32,replace=False)];s=[rows[int(i)] for i in ids]
        g=pooled(s,'g');a=pooled(s,'a');b=pooled(s,'b');ci_g=boot_delta(s,'g','zero',B=300,seed=int(rr.integers(1,1e9))) if False else None
        # Approximate dress gate power using paired bootstrap lower bounds.
        den=sum(x['den'] for x in s);nums=np.asarray([x['g'] for x in s]);dens=np.asarray([x['den'] for x in s]);
        vals=[];da=[];db=[]
        for __ in range(300):
            ii=rr.integers(0,len(s),len(s));dd=float(dens[ii].sum());vals.append(float(nums[ii].sum()/max(dd,1e-12)));da.append(float((nums[ii]-np.asarray([s[k]['a'] for k in range(len(s))])[ii]).sum()/max(dd,1e-12)));db.append(float((nums[ii]-np.asarray([s[k]['b'] for k in range(len(s))])[ii]).sum()/max(dd,1e-12)))
        h1+=int(g>=.35 and np.quantile(vals,.025)>=.25);h2a+=int((g-a)>=.10 and np.quantile(da,.025)>=.05);h2b+=int((g-b)>=.10 and np.quantile(db,.025)>=.05)
    whole.update({'dress_subsample_reps':reps,'h1_gap_gate_power':h1/reps,'h2_vs_b2a_gate_power':h2a/reps,'h2_vs_b2b_gate_power':h2b/reps})
    return whole

def main():
    out={'baseline_commit':'8ed59aade32eebbc8342f53c482ad304a16f3775','seeds':'non-dress only: 410000-419999','C1':c1(),'C2':c2(),'C3':c3(),'C4':c4(),'C5':c5(),'C6':c6()}
    out['verdicts']={
      'C1':'REPRODUCED' if all(vv['opens']==0 for vv in out['C1']['families'].values()) else 'NOT REPRODUCED',
      'C2':'REPRODUCED' if all(out['C2']['program_text'][k]=='TypeError' for k in ('moddiff','mean','prod','diff','modraw')) else 'NOT REPRODUCED',
      'C3':'REPRODUCED' if all(x['within_3pct_rate']>=.95 for x in out['C3']['r1_same_feature_ceiling'].values()) else 'NOT REPRODUCED',
      'C4':'REPRODUCED' if out['C4']['oracle_feature_max_abs_diff']['circular']<1e-12 and out['C4']['oracle_feature_max_abs_diff']['rank']<1e-12 else 'NOT REPRODUCED',
      'C5':'REPRODUCED' if out['C5']['stages']['challenge']==0 else 'NOT REPRODUCED',
      'C6':'REPRODUCED' if out['C6']['same_program']/96>.90 and out['C6']['h2_vs_b2a_gate_power']<.10 and out['C6']['h2_vs_b2b_gate_power']<.10 else 'NOT REPRODUCED'}
    path=ROOT/'results'/'v4.20-audit-addendum2.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
