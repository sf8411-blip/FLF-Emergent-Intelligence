#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib,numpy as np

def cp_upper(x,n,alpha=.05):
    if n<=0:return 1.0
    if x>=n:return 1.0
    lo,hi=0.,1.
    for _ in range(80):
        m=(lo+hi)/2;c=sum(math.comb(n,k)*m**k*(1-m)**(n-k) for k in range(x+1))
        if c>alpha:lo=m
        else:hi=m
    return (lo+hi)/2

def read_rows(root):
    rows=[]
    for f in sorted(pathlib.Path(root).rglob('*.jsonl')):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return rows

def r4(iteration,budget,root,out,checkpoint):
    rows=read_rows(root)
    if len(rows)!=64:raise SystemExit(f'need64 got{len(rows)}')
    fams={}
    for fam in ('circular','rank'):
        q=sorted([x for x in rows if x['family']==fam],key=lambda x:x['seed']);rate=float(np.mean([x['success'] for x in q]));oreg=float(np.mean([x['oracle_gain']>=.10 for x in q]));bok=all(int(x['candidate_evals'])==budget for x in q);fams[fam]={'n':len(q),'b2_success_rate':rate,'oracle_ge_.10_rate':oreg,'oracle_gain_min':float(np.min([x['oracle_gain'] for x in q])),'budget_ok':bok,'pass':bool(len(q)==32 and .10<=rate<=.60 and oreg>=.95 and bok)}
    same=float(np.mean([x['same_program'] for x in rows]));res={'iteration':iteration,'search_budget':budget,'families':fams,'R-F':{'n':64,'same_program_rate':same,'report_as_one_control':bool(same>.90),'pass':True},'pass':all(x['pass'] for x in fams.values())};pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(res,indent=2,sort_keys=True));pathlib.Path(checkpoint).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(checkpoint).write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in sorted(rows,key=lambda z:(z['family'],z['seed'])))+'\n');print(json.dumps(res,indent=2,sort_keys=True));raise SystemExit(0 if res['pass'] else 2)

def spurious(iteration,budget,root,out,checkpoint):
    rows=sorted({int(x['seed']):x for x in read_rows(root)}.values(),key=lambda x:int(x['seed']))
    if len(rows)!=300:raise SystemExit(f'need300 got{len(rows)}')
    pre=[x for x in rows if x['prechallenge']];chosen=pre[:30];fp=sum(bool(x.get('false_pass')) for x in chosen);n=len(chosen);rate=fp/max(n,1);up=cp_upper(fp,n);block=bool(chosen) and all(int(x['block_length'])>=int(x['acf_length']) for x in chosen);bok=all(int(x['candidate_evals'])==budget for x in rows);res={'iteration':iteration,'searched_seeds':300,'total_prechallenge':len(pre),'chosen_first_30_count':n,'chosen_seed_min':min((x['seed'] for x in chosen),default=None),'chosen_seed_max':max((x['seed'] for x in chosen),default=None),'false_passes':fp,'false_pass_rate':rate,'cp95_upper_one_sided':up,'block_ge_acf':block,'budget_ok':bok,'examples':chosen[:10],'pass':bool(n>=30 and rate<=.10 and up<=.10 and block and bok)};pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(res,indent=2,sort_keys=True,allow_nan=False));pathlib.Path(checkpoint).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(checkpoint).write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');print(json.dumps(res,indent=2,sort_keys=True));raise SystemExit(0 if res['pass'] else 2)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['r4','spurious']);p.add_argument('--iteration',type=int,required=True);p.add_argument('--budget',type=int,required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();(r4 if a.mode=='r4' else spurious)(a.iteration,a.budget,a.root,a.out,a.checkpoint)
if __name__=='__main__':main()
