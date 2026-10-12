#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib
import numpy as np

DRAFT_COMMIT='53b1da045214ea786ad4af7888918e46ac19f95f'
FAMILIES=('circular','rank','null','corrupt_circular','corrupt_rank','noise')
B=10000


def load_rows(root):
    out=[]
    for f in sorted(pathlib.Path(root).rglob('*.json')):
        txt=f.read_text().strip()
        if txt:out.append(json.loads(txt.splitlines()[-1]))
    return out


def ratio(rows,num='guided_num',den='oracle_den'):
    if not rows:return float('nan')
    d=sum(float(r[den]) for r in rows);return sum(float(r[num]) for r in rows)/d if d>1e-12 else float('nan')


def boot_ratio(rows,num='guided_num',den='oracle_den',seed=42001):
    if not rows:return [float('nan'),float('nan')]
    rr=np.random.default_rng(seed);n=len(rows);vals=[]
    for _ in range(B):
        s=[rows[int(i)] for i in rr.integers(0,n,n)];z=ratio(s,num,den)
        if np.isfinite(z):vals.append(z)
    if len(vals)<int(.95*B):return [float('nan'),float('nan')]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def boot_delta(rows,control_num,seed):
    if not rows:return [float('nan'),float('nan')]
    rr=np.random.default_rng(seed);n=len(rows);vals=[]
    for _ in range(B):
        s=[rows[int(i)] for i in rr.integers(0,n,n)];g=ratio(s,'guided_num','oracle_den');c=ratio(s,control_num,'oracle_den')
        if np.isfinite(g) and np.isfinite(c):vals.append(g-c)
    if len(vals)<int(.95*B):return [float('nan'),float('nan')]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def binom_cdf(x,n,p):
    return sum(math.comb(n,k)*(p**k)*((1-p)**(n-k)) for k in range(x+1))


def cp_upper_one_sided(x,n,alpha=.05):
    if n<=0:return float('nan')
    if x>=n:return 1.0
    lo,hi=0.0,1.0
    # Solve P_p[X <= x] = alpha; CDF decreases with p.
    for _ in range(90):
        mid=(lo+hi)/2
        if binom_cdf(x,n,mid)>alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2


def finite(x):return isinstance(x,(int,float)) and math.isfinite(float(x))
def gate_ge(point,ci,pt,lo):return finite(point) and finite(ci[0]) and point>=pt and ci[0]>=lo

def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True);p.add_argument('--out',required=True);p.add_argument('--raw',required=True);a=p.parse_args();rows=load_rows(a.rows)
    by={f:[r for r in rows if r['family']==f] for f in FAMILIES}
    counts={f:len(by[f]) for f in FAMILIES}
    if any(counts[f]!=32 for f in FAMILIES):raise SystemExit(f'exactly 32 dress rows/family required, got {counts}')
    # Unique seed/index check prevents accidental duplicate matrix artifacts.
    if len({(r['family'],r['index']) for r in rows})!=len(rows):raise SystemExit('duplicate family/index row')
    eligible=[r for f in ('circular','rank') for r in by[f] if bool(r['eligible'])]
    fam_eligible={f:[r for r in by[f] if bool(r['eligible'])] for f in ('circular','rank')}

    false_accept={}
    for f in ('null','corrupt_circular','corrupt_rank'):
        x=sum(bool(r['guided_accepted']) for r in by[f]);n=len(by[f]);rate=x/n;up=cp_upper_one_sided(x,n);false_accept[f]={'accepted':x,'n':n,'rate':rate,'cp95_upper_one_sided':up,'pass':bool(rate<=.05 and up<=.15)}

    g=ratio(eligible);gci=boot_ratio(eligible,seed=42011)
    tr=ratio(eligible,'transfer_num','transfer_den');trci=boot_ratio(eligible,'transfer_num','transfer_den',42012)
    ablrows=[r for r in eligible if bool(r['guided_accepted']) and float(r['guided_num'])>0 and r.get('ablation_num') is not None]
    A=ratio(ablrows,'ablation_num','guided_num');Aci=boot_ratio(ablrows,'ablation_num','guided_num',42013)

    h1_gates={
      'null_false_accept':false_accept['null']['pass'],
      'corrupt_circular_false_accept':false_accept['corrupt_circular']['pass'],
      'corrupt_rank_false_accept':false_accept['corrupt_rank']['pass'],
      'pooled_gap':gate_ge(g,gci,.35,.25),
      'transfer':gate_ge(tr,trci,.25,.15),
      'in_model_ablation':gate_ge(A,Aci,.20,.10),
    }
    h1=all(h1_gates.values())

    admitted=[r for r in eligible if bool(r['guided_accepted']) and r.get('functional_similarity') is not None]
    fs=float(np.median([r['functional_similarity'] for r in admitted])) if admitted else float('nan')
    ps=float(np.median([r['predictive_substitutability'] for r in admitted])) if admitted else float('nan')
    ga=ratio(eligible);a0=ratio(eligible,'b2a_num','oracle_den');b0=ratio(eligible,'b2b_num','oracle_den');da=ga-a0 if finite(ga) and finite(a0) else float('nan');db=ga-b0 if finite(ga) and finite(b0) else float('nan');daci=boot_delta(eligible,'b2a_num',42021);dbci=boot_delta(eligible,'b2b_num',42022)
    proposed=[r for r in eligible if r.get('guided_program') is not None]
    mdl_ok=all(r.get('guided_mdl_cost') is not None and abs(float(r['guided_mdl_cost'])-(1.0+.5*int(r['guided_constants'])))<1e-12 for r in proposed)
    h2_gates={'functional_similarity':finite(fs) and fs>=.90,'predictive_substitutability':finite(ps) and ps>=.90,'guided_vs_b2a':gate_ge(da,daci,.10,.05),'guided_vs_b2b':gate_ge(db,dbci,.10,.05),'mdl_constants_charged':bool(mdl_ok)}
    h2_status='PASS' if h1 and all(h2_gates.values()) else ('FAIL' if h1 else 'NOT_EVALUATED_CONDITIONAL_H1_FAIL')

    margin=bool(h1 and all(false_accept[f]['accepted']==0 for f in false_accept) and gate_ge(g,gci,.40,.30) and gate_ge(tr,trci,.30,.20) and gate_ge(A,Aci,.25,.15))
    family_diag={}
    for f in ('circular','rank'):
        rr=fam_eligible[f];family_diag[f]={'eligible':len(rr),'gap':ratio(rr),'gap_ci95':boot_ratio(rr,seed=42100+(0 if f=='circular' else 1)),'guided_accept_rate':sum(bool(x['guided_accepted']) for x in by[f])/32,'transfer':ratio(rr,'transfer_num','transfer_den'),'functional_similarity_median':float(np.median([x['functional_similarity'] for x in rr if x.get('functional_similarity') is not None])) if any(x.get('functional_similarity') is not None for x in rr) else float('nan')}
    noise={'trigger_open_rate':sum(bool(r['trigger_open']) for r in by['noise'])/32,'guided_accept_rate':sum(bool(r['guided_accepted']) for r in by['noise'])/32}
    b3={'pooled_gap':ratio(eligible,'b3_num','oracle_den'),'accepted_rate_nonnull':sum(bool(r['b3_accepted']) for r in eligible)/max(len(eligible),1),'all_rows_ran':all(bool(r['b3_ran']) for r in rows)}

    out={'draft_prereg_commit':DRAFT_COMMIT,'scope':'DEV_DRESS_ONLY','counts':counts,'eligible_nonnull':len(eligible),'false_acceptance':false_accept,'H1':{'status':'PASS' if h1 else 'FAIL','gates':h1_gates,'pooled_gap':g,'pooled_gap_ci95':gci,'transfer':tr,'transfer_ci95':trci,'ablation_damage':A,'ablation_ci95':Aci,'ablation_seed_count':len(ablrows)},'H2':{'status':h2_status,'gates':h2_gates if h1 else None,'functional_similarity_median':fs if h1 else None,'predictive_substitutability_median':ps if h1 else None,'guided_gap':ga if h1 else None,'b2a_gap':a0 if h1 else None,'b2b_gap':b0 if h1 else None,'delta_b2a':da if h1 else None,'delta_b2a_ci95':daci if h1 else None,'delta_b2b':db if h1 else None,'delta_b2b_ci95':dbci if h1 else None,'proposed_guided_count':len(proposed) if h1 else None,'admitted_eligible_count':len(admitted) if h1 else None},'frozen_run_margin_eligible':margin,'family_diagnostics':family_diag,'noise_diagnostic':noise,'b3_diagnostic':b3,'interpretation':'Bounded v4.20 DEV dress only; no frozen seed/T2 and no unrestricted math-genesis or AGI claim.'}
    pathlib.Path(a.raw).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.raw).write_text('\n'.join(json.dumps(r,sort_keys=True,allow_nan=False) for r in sorted(rows,key=lambda q:(q['family'],q['index'])))+'\n');pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(clean(out),indent=2,sort_keys=True,allow_nan=False));print(json.dumps(clean(out),indent=2,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
