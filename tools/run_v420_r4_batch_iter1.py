#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v420_repair_iter1 as r

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['circular','rank'],required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not 0<=a.batch<4:raise SystemExit('batch 0..3')
    base=523000+(0 if a.family=='circular' else 500);start=base+a.batch*8;rows=[]
    for i in range(8):
        s=start+i;env,y,h,m=r.make_world(a.family,s);bb=r.b0_and_oracle(env,y,h);den=bb['b0_test_rmse']-bb['b1_test_rmse'];og=den/max(bb['b0_test_rmse'],1e-12);aa,bb2,sigs,n=r.select_controls(env,y,bb,s+2001,r.SEARCH_BUDGET);frac=(bb['b0_test_rmse']-bb2[2]['test_rmse'])/max(den,1e-12);rows.append({'family':a.family,'seed':s,'oracle_gain':float(og),'b2_fraction':float(frac),'success':bool(frac>=.35),'candidate_evals':int(n),'same_program':aa[0].text()==bb2[0].text()})
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in rows)+'\n');print(json.dumps({'family':a.family,'batch':a.batch,'successes':sum(x['success'] for x in rows),'n':8}))
if __name__=='__main__':main()
