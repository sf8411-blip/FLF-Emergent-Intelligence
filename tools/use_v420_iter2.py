#!/usr/bin/env python3
from __future__ import annotations
import pathlib,runpy,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import flf_ecf
from flf_ecf import v420_repair as r0
from flf_ecf import v420_repair_iter2 as r2
for name in ('_valid_unary','_valid_binary','_init_param'):
    setattr(r2,name,getattr(r0,name))
sys.modules['flf_ecf.v420_repair']=r2
flf_ecf.v420_repair=r2
if len(sys.argv)<2:raise SystemExit('usage: use_v420_iter2.py SCRIPT [args...]')
script=sys.argv[1];sys.argv=[script]+sys.argv[2:];runpy.run_path(str(ROOT/script),run_name='__main__')
