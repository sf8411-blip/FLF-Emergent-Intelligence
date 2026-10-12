#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair_iter1 as r

DRESS=tuple(range(b,b+32) for b in (320000,321000,322000,323000,324000,325000,326000))
def nd(s):assert not any(int(s) in q for q in DRESS),s

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not 0<=a.batch<15:raise SystemExit('batch 0..14')
    rows=[];start=527000+a.batch*20
    for i in range(20):
        s=start+i;nd(s);env,y,h,m=r.make_world('spurious',s);bb=r.b0_and_oracle(env,y,h);sel,n=r.select_guided(env,y,bb,s+1001,r.SEARCH_BUDGET,False);_,e,z,fit,_=sel;ok,r2,gain=r.prechallenge(e,z,fit,bb)
        row={'seed':s,'prechallenge':bool(ok),'novelty_r2':float(r2),'validation_gain':float(gain),'program':e.text(),'candidate_evals':int(n),'rho':float(m['rho'])}
        if ok:
            pv,cg,d=r.challenge(e,'spurious',m,s+700001,B=999);row.update({'challenge_p':float(pv),'challenge_gain':float(cg),'acf_length':int(d['acf_length']),'block_length':int(d['block_length']),'false_pass':bool(pv<=r.CHALLENGE_P_MAX)})
        rows.append(row)
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');print(json.dumps({'batch':a.batch,'start':start,'n':len(rows),'prechallenge':sum(x['prechallenge'] for x in rows)},sort_keys=True))
if __name__=='__main__':main()
