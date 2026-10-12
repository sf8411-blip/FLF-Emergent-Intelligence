#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib

def cp_upper(x,n,alpha=.05):
    if n<=0:return 1.0
    if x>=n:return 1.0
    lo,hi=0.,1.
    for _ in range(80):
        m=(lo+hi)/2;c=sum(math.comb(n,k)*m**k*(1-m)**(n-k) for k in range(x+1))
        if c>alpha:lo=m
        else:hi=m
    return (lo+hi)/2

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();rows=[]
    for f in sorted(pathlib.Path(a.root).rglob('*.jsonl')):rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    rows=sorted({int(x['seed']):x for x in rows}.values(),key=lambda x:int(x['seed']))
    if len(rows)!=300:raise SystemExit(f'exactly 300 rows required, got {len(rows)}')
    pre=[x for x in rows if x['prechallenge']];chosen=pre[:30];fp=sum(bool(x.get('false_pass')) for x in chosen);n=len(chosen);rate=fp/max(n,1);up=cp_upper(fp,n);block_ok=bool(chosen) and all(int(x['block_length'])>=int(x['acf_length']) for x in chosen);budget=all(int(x['candidate_evals'])==1024 for x in rows);ok=bool(n>=30 and rate<=.10 and up<=.10 and block_ok and budget)
    out={'iteration':2,'searched_seeds':300,'total_prechallenge':len(pre),'chosen_first_30_count':n,'false_passes':fp,'false_pass_rate':rate,'cp95_upper_one_sided':up,'block_ge_acf':block_ok,'budget_all_1024':budget,'chosen_seed_min':min((x['seed'] for x in chosen),default=None),'chosen_seed_max':max((x['seed'] for x in chosen),default=None),'examples':chosen[:10],'pass':ok};pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));pathlib.Path(a.checkpoint).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.checkpoint).write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
