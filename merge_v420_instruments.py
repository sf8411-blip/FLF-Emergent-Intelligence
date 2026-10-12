#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v420_dev import bootstrap_ci,MDL_CONSTANT_COST,NOVELTY_R2_MAX,VALIDATION_GAIN_MIN,CHALLENGE_P_MAX
p=argparse.ArgumentParser();p.add_argument('--instrument-dir',required=True);p.add_argument('--noise-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
def rows(d):
 out=[]
 for f in sorted(pathlib.Path(d).glob('*.json')):
  txt=f.read_text().strip()
  if txt:out.append(json.loads(txt.splitlines()[-1]))
 return out
r=rows(a.instrument_dir);n=rows(a.noise_dir)
if len(r)!=32:raise SystemExit(f'need exactly 32 instrument DEV rows, got {len(r)}')
if len(n)!=200:raise SystemExit(f'need exactly 200 noise DEV rows, got {len(n)}')
it={x['iteration'] for x in r+n}
if len(it)!=1:raise SystemExit('mixed DEV iterations are invalid')
iteration=next(iter(it));out={'iteration':iteration,'n_instrument':len(r),'n_noise':len(n),'criteria':{}}
rate=np.mean([x['R1_relative_gap']<=.03 for x in r]);out['criteria']['R1']={'within_3pct_rate':float(rate),'threshold':.95,'pass':bool(rate>=.95)}
r2={}
for fam in ('corrupt_circular','corrupt_rank'):
 v=np.asarray([x[f'R2_{fam}_oracle_gain'] for x in r]);ci=bootstrap_ci(v);ok=abs(v.mean())<=.01 and ci[0]<=0<=ci[1];r2[fam]={'mean':float(v.mean()),'ci95':ci,'pass':bool(ok)}
out['criteria']['R2']={'families':r2,'pass':all(x['pass'] for x in r2.values())}
grid=sum(int(x['R3_grid_matches']) for x in r)
continuous=(iteration<4) or all(x.get('R3_constant_search')=='continuous_1d_refinement' for x in r)
out['criteria']['R3']={'grid_matches':grid,'constant_mdl_cost':MDL_CONSTANT_COST,'continuous_1d_search':bool(continuous),'pass':bool(grid==0 and MDL_CONSTANT_COST==.5 and continuous)}
r4={}
for fam in ('circular','rank'):
 og=np.asarray([x[f'R4_{fam}_oracle_gain'] for x in r]);sr=np.mean([x[f'R4_{fam}_B2_success'] for x in r]);ok=np.mean(og>=.10)>=.90 and np.min(og)>=0 and .1<=sr<=.6;r4[fam]={'oracle_mean':float(og.mean()),'oracle_ge_.10_rate':float(np.mean(og>=.10)),'oracle_min':float(og.min()),'B2_success_rate':float(sr),'pass':bool(ok)}
out['criteria']['R4']={'families':r4,'pass':all(x['pass'] for x in r4.values())}
fr=np.mean([x['open'] for x in n]);out['criteria']['R5']={'false_open_rate':float(fr),'opened':int(sum(x['open'] for x in n)),'n':200,'pass':bool(fr<=.02)}
fa=np.mean([x['R6_null_accepted'] for x in r])
r6={'null_accept_rate':float(fa),'novelty_r2_max':NOVELTY_R2_MAX,'validation_gain_min':VALIDATION_GAIN_MIN,'challenge_p_max':CHALLENGE_P_MAX}
if iteration>=4:
 pos={}
 for fam in ('circular','rank'):
  ok=[bool(x.get(f'R6_{fam}_same_meta',False)) and float(x.get(f'R6_{fam}_positive_challenge_p',1.0))<=CHALLENGE_P_MAX for x in r]
  ratep=float(np.mean(ok));maxp=max(float(x.get(f'R6_{fam}_positive_challenge_p',1.0)) for x in r)
  pos[fam]={'pass_rate':ratep,'required_rate':.95,'max_p':maxp,'pass':bool(ratep>=.95)}
 r6['positive_same_world_challenge']=pos
 r6['pass']=bool(fa<=.05 and all(z['pass'] for z in pos.values()))
else:r6['pass']=bool(fa<=.05)
out['criteria']['R6']=r6
# R7/R8 are executable semantic instruments. Their validity is established by
# tests/test_v420_eval.py and tests/test_v420_protocol.py in the same HEAD CI;
# never manufacture an overall pass from declarative JSON flags alone.
out['criteria']['R7']={'note':'Requires green executable CI: pooled ratio-of-sums + seed bootstrap; in-model block ablation without refit; world-trace similarity + predictive substitutability.'}
out['criteria']['R8']={'note':'Requires green executable CI: B3 always searches and B2a/B2b share the same random grammar/bank with their specified selectors.'}
out['instrument_valid_numeric_R1_R6']=bool(all(out['criteria'][k]['pass'] for k in ('R1','R2','R3','R4','R5','R6')))
out['instrument_valid']=False
out['note']='Overall Step-2 validity may be declared only after numeric R1-R6 pass and executable R7/R8/protocol CI is green on the same HEAD.'
path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2))
