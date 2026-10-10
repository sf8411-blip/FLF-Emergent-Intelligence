#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from flf_ecf import v419_runner
from flf_ecf.v419_runner import run_one,run_control

PREREG_PATH=ROOT/'docs/v4.19_preregistration.md'
PREREG_SHA256=hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
PREREG_BLOB_SHA=os.environ.get('V419_PREREG_BLOB_SHA','UNKNOWN')
v419_runner.set_integrity(PREREG_SHA256,PREREG_BLOB_SHA)

TIERS={'T0':(200,2),'T1':(2000,20),'T2':(20000,24)}

def atomic_append_jsonl(path: pathlib.Path, rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=[]
    if path.exists(): existing=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    keys={(r.get('family'),r.get('seed')) for r in existing}
    merged=existing+[r for r in rows if (r.get('family'),r.get('seed')) not in keys]
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in merged))
    os.replace(tmp,path)

def parse_seed_spec(s):
    out=[]
    for p in s.split(','):
        if '-' in p:
            a,b=map(int,p.split('-',1));out.extend(range(a,b+1))
        elif p.strip():out.append(int(p))
    return out

p=argparse.ArgumentParser()
p.add_argument('--tier',choices=TIERS,default='T0');p.add_argument('--budget',type=int)
p.add_argument('--seeds',default='');p.add_argument('--shard',default='0/1');p.add_argument('--resume',action='store_true')
p.add_argument('--out',default='results/v4.19-checkpoint.jsonl');p.add_argument('--reachability',default='results/v4.19-reachability.json');p.add_argument('--code-sha',default=os.environ.get('GITHUB_SHA','UNKNOWN'))
a=p.parse_args(); prereg_budget,count=TIERS[a.tier]
budget=a.budget if a.budget is not None else prereg_budget
if budget<prereg_budget: raise SystemExit(f'Budget {budget} is below preregistered {prereg_budget}; refusing silent downscale.')
frozen=a.tier=='T2'
if a.tier in ('T1','T2'):
    rp=ROOT/a.reachability
    if not rp.exists(): raise SystemExit('Reachability audit missing; T1/T2 blocked.')
    audit=json.loads(rp.read_text())
    if not audit.get('pass') or audit.get('preregistration_sha256')!=PREREG_SHA256: raise SystemExit('Reachability audit invalid; T1/T2 blocked.')
if a.seeds: seeds=parse_seed_spec(a.seeds)
elif a.tier=='T0': seeds=[1400,1401]
elif a.tier=='T1': seeds=list(range(1400,1420))
else: seeds=[]
i,nsh=map(int,a.shard.split('/'))
rows=[]
if frozen:
    tasks=[('circular',s) for s in range(61000,61024)]+[('rank_interaction',s) for s in range(62000,62024)]+[('null',s) for s in range(63000,63024)]+[('actuator_shuffle',s) for s in range(71000,71024)]+[('noise',s) for s in range(71100,71124)]
else:
    tasks=[(f,s) for f in ('circular','rank_interaction') for s in seeds]
    if a.tier=='T1': tasks += [('null',1400+j) for j in range(10)]
tasks=[t for j,t in enumerate(tasks) if j%nsh==i]
outpath=ROOT/a.out
completed=set()
if outpath.exists():
    if not a.resume: raise SystemExit(f'{a.out} exists; use --resume to continue idempotently.')
    completed={(r.get('family'),r.get('seed')) for r in (json.loads(x) for x in outpath.read_text().splitlines() if x.strip())}
tasks=[t for t in tasks if t not in completed]
for family,seed in tasks:
    ss=(7000+(seed%1000)) if not frozen else 600000+seed
    if family in ('circular','rank_interaction'):
        r=run_one(family,seed,budget,ss,a.tier,frozen,71000+(seed%1000),a.code_sha)
    else:r=run_control(family,seed,budget,ss,a.tier,frozen,a.code_sha)
    r['shard']=a.shard
    rows.append(r);atomic_append_jsonl(ROOT/a.out,[r])
print(json.dumps({'tier':a.tier,'budget':budget,'shard':a.shard,'completed':len(rows),'out':a.out,'preregistration_sha256':PREREG_SHA256},indent=2))
