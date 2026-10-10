#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
PREREG=hashlib.sha256((ROOT/'docs/v4.19_preregistration.md').read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',default='results/v4.19-reachability.json');a=p.parse_args()
parts=[json.loads(pathlib.Path(x).read_text()) for x in a.inputs]
if any(x.get('preregistration_sha256')!=PREREG for x in parts):raise SystemExit('reachability preregistration hash mismatch')
fams=[]
for x in parts:fams.extend(x.get('families',[]))
by={f['family']:f for f in fams}
needed=('circular','rank_interaction');complete=all(k in by for k in needed)
ok=complete and all(by[k].get('pass') for k in needed)
report={'preregistration_sha256':PREREG,'preregistration_git_blob_sha':next((x.get('preregistration_git_blob_sha') for x in parts if x.get('preregistration_git_blob_sha')),None),
        'code_commit_shas':sorted(set(x.get('code_commit_sha','UNKNOWN') for x in parts)),'diagnostic_count':12,
        'families':[by[k] for k in needed if k in by],'pass':bool(ok)}
out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out);print(json.dumps(report,indent=2))
if not ok:sys.exit(2)
