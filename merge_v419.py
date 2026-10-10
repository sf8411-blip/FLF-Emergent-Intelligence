#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pathlib
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent
PREREG=hashlib.sha256((ROOT/'docs/v4.19_preregistration.md').read_bytes()).hexdigest()

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
complete=True; gates=[]
for fam in ('circular','rank_interaction'):
    q=[r for r in rows if r['family']==fam and r.get('frozen_test')]
    if len(q)<24:complete=False;continue
    g=np.array([r['gap_closed'] for r in q]);tr=np.array([r['transfer_gap_closed'] for r in q]);ab=np.array([r['ablation_damage'] for r in q])
    md_gain=np.array([r['mdl_improvement'] for r in q]); md_margin=np.array([r['mdl_improvement']-.02*r['b0_residual_description_length'] for r in q])
    rd=np.array([r['gap_closed']-r['random_gap_closed'] for r in q]);sd=np.array([r['gap_closed']-r['shuffled_gap_closed'] for r in q])
    sig=[r['functional_signature'] for r in q if r.get('accepted') and r.get('functional_signature')]
    sims=[similarity(sig[i],sig[j]) for i in range(len(sig)) for j in range(i+1,len(sig))]
    vals={'n':len(q),'gap_closed_mean':float(g.mean()),'gap_closed_ci':boot_ci(g),'transfer_mean':float(tr.mean()),'transfer_ci':boot_ci(tr),
          'ablation_mean':float(ab.mean()),'ablation_ci':boot_ci(ab),'mdl_improvement_mean':float(md_gain.mean()),'mdl_margin_ci':boot_ci(md_margin),
          'residual_minus_random_mean':float(rd.mean()),'residual_minus_random_ci':boot_ci(rd),'residual_minus_shuffled_mean':float(sd.mean()),'residual_minus_shuffled_ci':boot_ci(sd),
          'functional_similarity_median':float(np.median(sims)) if sims else 0.0,'functional_similarity_ci':boot_ci(sims,'median') if sims else [0,0]}
    vals['G1']=vals['gap_closed_mean']>=.35 and vals['gap_closed_ci'][0]>=.25
    vals['G2']=vals['transfer_mean']>=.25 and vals['transfer_ci'][0]>=.15
    vals['G3']=vals['ablation_mean']>=.20 and vals['ablation_ci'][0]>=.10
    vals['G4']=vals['functional_similarity_median']>=.90 and vals['functional_similarity_ci'][0]>=.85
    vals['G5']=vals['mdl_improvement_mean']>0 and vals['mdl_margin_ci'][0]>=0.0
    vals['G6']=vals['residual_minus_random_mean']>=.10 and vals['residual_minus_random_ci'][0]>=.05
    vals['G7']=vals['residual_minus_shuffled_mean']>=.10 and vals['residual_minus_shuffled_ci'][0]>=.05 and all(not r.get('disabled_accepted') for r in q)
    summary['families'][fam]=vals
    gates += [vals[f'G{i}'] for i in range(1,8)]
null=[r for r in rows if r['family']=='null' and r.get('frozen_test')]
if len(null)<24:complete=False
else:
    k=sum(bool(r['accepted']) for r in null);rate=k/len(null);up=cp_upper(k,len(null));ok=rate<=.05 and up<=.15
    summary['null']={'n':len(null),'accept_rate':rate,'cp95_upper':up,'G8':ok};gates.append(ok)
for ctrl in ('actuator_shuffle','noise'):
    q=[r for r in rows if r['family']==ctrl and r.get('frozen_test')]
    if len(q)<24:complete=False;continue
    k=sum(bool(r['accepted']) for r in q);rate=k/len(q);up=cp_upper(k,len(q));ok=rate<=.10 and up<=.20
    summary[ctrl]={'n':len(q),'accept_rate':rate,'cp95_upper':up,'G9':ok};gates.append(ok)
reach_path=ROOT/'results/v4.19-reachability.json'
if not reach_path.exists():
    complete=False;summary['reachability']={'pass':False,'reason':'missing'}
else:
    reach=json.loads(reach_path.read_text());ok=bool(reach.get('pass')) and reach.get('preregistration_sha256')==PREREG
    summary['reachability']=reach;summary['gates']['G10']=ok;gates.append(ok)
summary['gates']['all_non_reachability']=all(gates[:-1]) if gates else False
if complete:summary['verdict']='PASS' if all(gates) else 'FAIL'
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(out);print(json.dumps(summary,indent=2))
