#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib
import numpy as np

PREREG='c870ac245b5c4ed1d70c32e1ba97dd87e48106d7fbb6807ed167bca9d2bf4e19'

def boot_ci(x,stat='mean',B=20000,seed=31991):
    x=np.asarray(x,float);r=np.random.default_rng(seed);vals=[]
    for _ in range(B):
        s=x[r.integers(0,len(x),len(x))];vals.append(float(np.mean(s) if stat=='mean' else np.median(s)))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def binom_cdf(k,n,p):
    from math import comb
    return sum(comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(k+1))

def cp_upper(k,n,alpha=.05):
    if k==n:return 1.0
    lo,hi=0.0,1.0;target=alpha/2
    for _ in range(80):
        mid=(lo+hi)/2
        if binom_cdf(k,n,mid)>target:lo=mid
        else:hi=mid
    return (lo+hi)/2

def similarity(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);n=min(len(a),len(b));a=a[:n];b=b[:n]
    if n<3:return 0.0
    return abs(float(np.corrcoef(a,b)[0,1]))

p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
rows=[]
for fn in a.inputs:
    for line in pathlib.Path(fn).read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
if any(r.get('preregistration_sha256')!=PREREG for r in rows):raise SystemExit('preregistration hash mismatch')
summary={'preregistration_sha256':PREREG,'n_rows':len(rows),'families':{},'gates':{},'verdict':'INCONCLUSIVE (under-powered)'}
complete=True;gates=[]
for fam in ('circular','rank_interaction'):
    q=[r for r in rows if r['family']==fam and r.get('frozen_test')]
    if len(q)<24:complete=False;continue
    g=np.array([r['gap_closed'] for r in q]);tr=np.array([r['transfer_gap_closed'] for r in q]);ab=np.array([r['ablation_damage'] for r in q])
    md=np.array([r['mdl_improvement']/max(abs(r['b0_residual_description_length']),1e-9) for r in q])
    rd=np.array([r['gap_closed']-r['random_gap_closed'] for r in q]);sd=np.array([r['gap_closed']-r['shuffled_gap_closed'] for r in q])
    sig=[r['functional_signature'] for r in q if r.get('accepted') and r.get('functional_signature')]
    sims=[similarity(sig[i],sig[j]) for i in range(len(sig)) for j in range(i+1,len(sig))]
    vals={'gap_closed_mean':float(g.mean()),'gap_closed_ci':boot_ci(g),'transfer_mean':float(tr.mean()),'transfer_ci':boot_ci(tr),
          'ablation_mean':float(ab.mean()),'ablation_ci':boot_ci(ab),'mdl_ratio_mean':float(md.mean()),'mdl_ratio_ci':boot_ci(md),
          'residual_minus_random_mean':float(rd.mean()),'residual_minus_random_ci':boot_ci(rd),'residual_minus_shuffled_mean':float(sd.mean()),'residual_minus_shuffled_ci':boot_ci(sd),
          'functional_similarity_median':float(np.median(sims)) if sims else 0.0,'functional_similarity_ci':boot_ci(sims,'median') if sims else [0,0]}
    summary['families'][fam]=vals
    gates += [vals['gap_closed_mean']>=.35 and vals['gap_closed_ci'][0]>=.25,
              vals['transfer_mean']>=.25 and vals['transfer_ci'][0]>=.15,
              vals['ablation_mean']>=.20 and vals['ablation_ci'][0]>=.10,
              vals['functional_similarity_median']>=.90 and vals['functional_similarity_ci'][0]>=.85,
              vals['mdl_ratio_mean']>0 and vals['mdl_ratio_ci'][0]>=.02,
              vals['residual_minus_random_mean']>=.10 and vals['residual_minus_random_ci'][0]>=.05,
              vals['residual_minus_shuffled_mean']>=.10 and vals['residual_minus_shuffled_ci'][0]>=.05]
null=[r for r in rows if r['family']=='null' and r.get('frozen_test')]
if len(null)<24:complete=False
else:
    k=sum(bool(r['accepted']) for r in null);rate=k/len(null);up=cp_upper(k,len(null));summary['null']={'accept_rate':rate,'cp95_upper':up};gates.append(rate<=.05 and up<=.15)
if complete:summary['verdict']='PASS' if all(gates) else 'FAIL'
pathlib.Path(a.out).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
