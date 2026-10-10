#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
PREREG=hashlib.sha256((ROOT/'docs/v4.19_preregistration.md').read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',default='results/v4.19-t1-summary.json');a=p.parse_args()
rows=[]
for fn in a.inputs:
    for line in pathlib.Path(fn).read_text().splitlines():
        if line.strip():rows.append(json.loads(line))
if any(r.get('preregistration_sha256')!=PREREG for r in rows):raise SystemExit('pilot preregistration hash mismatch')
out={'preregistration_sha256':PREREG,'tier':'T1','inferential_verdict':'NONE — development pilot only','n_rows':len(rows),'families':{}}
for fam in sorted(set(r['family'] for r in rows)):
    q=[r for r in rows if r['family']==fam];d={'n':len(q),'acceptance_rate':statistics.mean(bool(r.get('accepted')) for r in q),'mean_trigger':statistics.mean(float(r.get('trigger_score',0)) for r in q),'mean_candidate_evaluations':statistics.mean(int(r.get('candidate_evaluations',0)) for r in q)}
    if fam in ('circular','rank_interaction'):
        d.update({'mean_b0_error':statistics.mean(r['b0_error'] for r in q),'mean_b1_error':statistics.mean(r['b1_error'] for r in q),'mean_gap_closed':statistics.mean(r['gap_closed'] for r in q),'mean_random_gap_closed':statistics.mean(r['random_gap_closed'] for r in q),'mean_shuffled_gap_closed':statistics.mean(r['shuffled_gap_closed'] for r in q)})
    out['families'][fam]=d
path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path);print(json.dumps(out,indent=2))
