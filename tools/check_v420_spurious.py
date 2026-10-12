#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair as r
DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nd(s):assert not any(int(s) in q for q in DRESS)
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
    p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();it=a.iteration;B=500000+it*20000;rows=[];cp=pathlib.Path(a.checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
    for i in range(300):
        s=B+7000+i;nd(s);env,y,h,m=r.make_world('spurious',s);bb=r.b0_and_oracle(env,y,h);sel,n=r.select_guided(env,y,bb,s+1001,r.SEARCH_BUDGET,False);_,e,z,fit,_=sel;ok,r2,gain=r.prechallenge(e,z,fit,bb)
        rec={'iteration':it,'seed':s,'rho':m['rho'],'prechallenge':bool(ok),'novelty_r2':r2,'validation_gain':gain,'program':e.text(),'candidate_evals':n}
        if ok:
            pv,cg,d=r.challenge(e,'spurious',m,s+700001,B=999);rec.update({'challenge_p':pv,'challenge_gain':cg,'acf_length':d['acf_length'],'block_length':d['block_length'],'false_pass':bool(pv<=r.CHALLENGE_P_MAX)})
        rows.append(rec);cp.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n')
        if sum(x['prechallenge'] for x in rows)>=30:break
    pre=[x for x in rows if x['prechallenge']];fp=sum(bool(x.get('false_pass')) for x in pre);up=cp_upper(fp,len(pre));block_ok=bool(pre) and all(x['block_length']>=x['acf_length'] for x in pre);rate=fp/max(len(pre),1);ok=bool(len(pre)>=30 and rate<=.10 and up<=.10 and block_ok and all(x['candidate_evals']==r.SEARCH_BUDGET for x in rows))
    out={'iteration':it,'searched_seeds':len(rows),'prechallenge_count':len(pre),'false_passes':fp,'false_pass_rate':rate,'cp95_upper_one_sided':up,'block_ge_acf':block_ok,'search_budget':r.SEARCH_BUDGET,'examples':pre[:10],'pass':ok};pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False));raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
