#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();rows=[]
    for f in sorted(pathlib.Path(a.root).rglob('*.jsonl')):rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if len(rows)!=64:raise SystemExit(f'need 64 rows, got {len(rows)}')
    fams={}
    for fam in ('circular','rank'):
        q=sorted([x for x in rows if x['family']==fam],key=lambda x:x['seed']);rate=float(np.mean([x['success'] for x in q]));oreg=float(np.mean([x['oracle_gain']>=.10 for x in q]));budget=all(int(x['candidate_evals'])==1024 for x in q);fams[fam]={'b2_success_rate':rate,'oracle_ge_.10_rate':oreg,'oracle_gain_min':float(np.min([x['oracle_gain'] for x in q])),'budget_all_1024':budget,'pass':bool(len(q)==32 and .10<=rate<=.60 and oreg>=.95 and budget)}
    same=float(np.mean([x['same_program'] for x in rows]));out={'iteration':2,'search_budget':1024,'families':fams,'R-F':{'n':64,'same_program_rate':same,'report_as_one_control':bool(same>.90),'pass':True},'pass':all(x['pass'] for x in fams.values())};pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True));pathlib.Path(a.checkpoint).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.checkpoint).write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in sorted(rows,key=lambda z:(z['family'],z['seed'])))+'\n');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 2)
if __name__=='__main__':main()
