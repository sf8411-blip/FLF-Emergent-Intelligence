#!/usr/bin/env python3
from __future__ import annotations
import argparse, inspect, json, pathlib, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v419_worlds import make_world, hidden_feature, reference_expr
from flf_ecf.v419_dsl import Expr, var, evaluate, l0_expressions, mdl_cost
from flf_ecf.v419_learner import trigger_components, random_proposal
from flf_ecf.v419_v418_adapter import select_three
import flf_ecf.v419_runner as v419_runner
TRIGGER_THRESHOLD=2.5

def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
 if m.sum()<3:return 0.0
 a=a[m];b=b[m]
 return 0.0 if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def split_args(s):
 out=[];d=0;st=0
 for i,ch in enumerate(s):
  if ch=='(':d+=1
  elif ch==')':d-=1
  elif ch==',' and d==0:out.append(s[st:i].strip());st=i+1
 out.append(s[st:].strip());return out

def parse_expr(text):
 text=text.strip()
 if '(' not in text:return var(text)
 p=text.find('(');head=text[:p];inside=text[p+1:-1]
 if '[' in head:
  op=head[:head.index('[')];raw=head[head.index('[')+1:-1]
  try:param=int(raw)
  except ValueError:param=float(raw)
 else:op=head;param=None
 return Expr(op,tuple(parse_expr(x) for x in split_args(inside)),param)

def world(fam,seed):
 env,u,y=make_world(fam,int(seed));return {'env':env,'u':u,'y':y}

def augment(w):
 env={k:np.asarray(v,float) for k,v in w['env'].items()};u=np.asarray(w['u'],float)
 for j in range(u.shape[1]):env[f'u{j}']=u[:,j]
 return env

def r2_l0(expr,env):
 cols=[]
 for e in l0_expressions(sorted(env)):
  try:cols.append(np.asarray(evaluate(e,env),float))
  except Exception:pass
 y=np.asarray(evaluate(expr,env),float);X=np.column_stack(cols);m=np.isfinite(y)&np.all(np.isfinite(X),axis=1)
 X=np.column_stack([np.ones(m.sum()),X[m]]);yy=y[m];b=np.linalg.lstsq(X,yy,rcond=None)[0];p=X@b
 return 1-float(np.sum((yy-p)**2))/max(float(np.sum((yy-yy.mean())**2)),1e-12)

def pairmed(vs):
 z=[abs(corr(vs[i],vs[j])) for i in range(len(vs)) for j in range(i+1,len(vs))]
 return float(np.median(z)) if z else 0.0

def approx(x,t,tol):return abs(float(x)-float(t))<=tol

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',default='results/v4.19-t2-raw.jsonl');ap.add_argument('--out',default='results/v4.20-audit-v419.json');ap.add_argument('--noise-sims',type=int,default=10000);a=ap.parse_args()
 rows=[json.loads(x) for x in (ROOT/a.raw).read_text().splitlines() if x.strip()];by={f:[r for r in rows if r['family']==f] for f in {r['family'] for r in rows}};C={}
 # a
 cc=[];ac=[];oc=[];or2=[]
 for r in by['circular']:
  w=world('circular',r['seed']);cc.append(abs(corr(w['y'],hidden_feature('circular',w['env']))))
 for r in by['actuator_shuffle']:
  w=world('actuator_shuffle',r['seed']);h=hidden_feature('circular',w['env']);ac.append(abs(corr(w['y'],h)))
  if r.get('accepted') and r.get('operator'):
   e=parse_expr(r['operator']);env=augment(w);oc.append(abs(corr(evaluate(e,env),h)));or2.append(r2_l0(e,env))
 ws=inspect.getsource(make_world).replace(' ','')
 d={'circular_y_hidden_corr_mean':float(np.mean(cc)),'actuator_shuffle_y_hidden_corr_mean':float(np.mean(ac)),'accepted':len(oc),'operator_hidden_corr_median':float(np.median(oc)),'operator_l0_r2_median':float(np.median(or2)),'code_only_permuted_action':"action=r.permutation(action)" in ws and 'y=1.8*h+.08*action' in ws}
 C['a']={'status':'REPRODUCED' if len(oc)==23 and approx(np.mean(cc),.887,.04) and approx(np.mean(ac),.878,.04) and approx(np.median(oc),.86,.12) and approx(np.median(or2),.09,.15) and d['code_only_permuted_action'] else 'NOT REPRODUCED','diagnostics':d}
 # b
 rs=[];ops=[]
 for r in by['null']:
  if r.get('accepted') and r.get('operator'):
   w=world('null',r['seed']);e=parse_expr(r['operator']);rs.append(r2_l0(e,augment(w)));ops.append(r['operator'])
 nullmin=min(float(r['trigger_score']) for r in by['null']);ss=inspect.getsource(select_three);accept_src=inspect.getsource(v419_runner._candidate_accept)
 d={'accepted':len(rs),'l0_r2_median':float(np.median(rs)) if rs else None,'trigger_min':nullmin,'example':ops[0] if ops else None,'three_slot_selector':'need=3-len(req)' in ss.replace(' ',''),'novelty_gate_present':'novel' in accept_src.lower() or 'r2' in accept_src.lower()}
 C['b']={'status':'REPRODUCED' if len(rs)==14 and np.median(rs)>=.99 and nullmin>=9 and d['three_slot_selector'] and not d['novelty_gate_present'] else 'NOT REPRODUCED','diagnostics':d}
 # c
 nop=sum(float(r['trigger_score'])>=2.5 for r in by['noise']);rng=np.random.default_rng(44001);opens=0;n=406
 for _ in range(a.noise_sims):
  rr=rng.normal(size=n);uu=rng.normal(size=(n,2));opens+=trigger_components(rr,uu)[0]>=2.5
 pp=opens/a.noise_sims;d={'threshold':2.5,'sim_iid_open_rate':pp,'noise_open_count':nop,'noise_n':len(by['noise'])}
 C['c']={'status':'REPRODUCED' if nop==12 and approx(pp,.51,.08) else 'NOT REPRODUCED','diagnostics':d}
 # d
 ca=[r for r in by['circular'] if r.get('accepted') and r.get('operator')];iid=pairmed([r['functional_signature'] for r in ca]);vec=[];hc=[];common=[r['seed'] for r in by['circular'][:8]]
 for r in ca:
  e=parse_expr(r['operator']);pieces=[]
  for s in common:
   w=world('circular',s);z=np.asarray(evaluate(e,augment(w)),float);pieces.append(np.nan_to_num((z-np.nanmean(z))/(np.nanstd(z)+1e-12)))
  vec.append(np.concatenate(pieces));w=world('circular',r['seed']);hc.append(abs(corr(evaluate(e,augment(w)),hidden_feature('circular',w['env']))))
 wm=pairmed(vec);hm=float(np.median(hc));d={'accepted':len(ca),'iid_signature_pairwise_median':iid,'world_trace_pairwise_median':wm,'hidden_feature_corr_median':hm}
 C['d']={'status':'REPRODUCED' if len(ca)==22 and approx(iid,.164,.05) and approx(wm,.877,.15) and approx(hm,.963,.10) else 'NOT REPRODUCED','diagnostics':d}
 # e
 src=inspect.getsource(random_proposal);rg=[float(r['random_gap_closed']) for r in by['circular']];nn=by['circular']+by['rank_interaction'];sz=all(abs(float(r['shuffled_gap_closed']))<1e-12 for r in nn);g3=all(abs(float(r['ablation_damage'])-((r['b0_error']-r['l1_error'])/max(r['l1_error'],1e-12)))<1e-10 for r in nn)
 d={'random_gap_mean':float(np.mean(rg)),'random_positive_count':sum(x>0 for x in rg),'shuffled_zero_48':sz,'g3_transform_exact':g3,'random_uses_true_residual':"np.corrcoef(zn,r)" in src,'pool_not_expanded':"pool=[var(v) for v in variables]" in src and 'pool.append' not in src}
 C['e']={'status':'REPRODUCED' if np.mean(rg)>.75 and len(rg)==24 and all(x>0 for x in rg) and sz and g3 and d['random_uses_true_residual'] and d['pool_not_expanded'] else 'NOT REPRODUCED','diagnostics':d}
 # f
 texts=[r['operator'] for r in ca];mod=sum('modulo[1.0]' in x for x in texts);cs=sum('cumsum' in x for x in texts);ref=reference_expr('rank_interaction').text();ms=inspect.getsource(mdl_cost).replace(' ','')
 d={'modulo_1_count':mod,'accepted':len(texts),'cumsum_count':cs,'rank_reference':ref,'mdl_zero_constants':'learned_constants=0' in ms}
 C['f']={'status':'REPRODUCED' if mod==len(texts)==22 and cs==0 and all(q in ref for q in ('rank01[9]','threshold_gt[0.75]','filter_mean[3]')) and d['mdl_zero_constants'] else 'NOT REPRODUCED','diagnostics':d}
 # g
 rk=by['rank_interaction'];og=[(r['b0_error']-r['b1_error'])/max(r['b0_error'],1e-12) for r in rk];gg=[float(r['gap_closed']) for r in rk];im=int(np.argmax(gg));pool=sum(r['b0_error']-r['l1_error'] for r in rk)/max(sum(r['b0_error']-r['b1_error'] for r in rk),1e-12)
 d={'negative_oracle_gain_count':sum(x<0 for x in og),'mean_gap':float(np.mean(gg)),'max_gap_seed':rk[im]['seed'],'max_gap':gg[im],'max_gap_oracle_gain':og[im],'pooled_ratio':float(pool)}
 C['g']={'status':'REPRODUCED' if sum(x<0 for x in og)==4 and approx(np.mean(gg),.34,.03) and approx(og[im],.008,.02) and approx(gg[im],7.77,1.0) and approx(pool,.048,.03) else 'NOT REPRODUCED','diagnostics':d}
 # h
 p=ROOT/'results/v4.19-t1-summary.json'
 if p.exists():
  j=json.loads(p.read_text());f=j['families'];nu=f['null']['acceptance_rate'];ra=f['rank_interaction']['acceptance_rate'];rr=f['rank_interaction']['mean_random_gap_closed'];gu=f['rank_interaction']['mean_gap_closed'];ok=approx(nu,.5,.02) and approx(ra,.2,.02) and rr>gu;d={'null_acceptance':nu,'rank_acceptance':ra,'random_gap':rr,'guided_gap':gu}
 else:ok=False;d={'reason':'Exact committed T1 pilot summary/raw bytes are absent from the closed branch; historical narrative is insufficient for independent reproduction.'}
 C['h']={'status':'REPRODUCED' if ok else 'NOT REPRODUCED','diagnostics':d}
 out={'source_raw':a.raw,'n_rows':len(rows),'claims':C,'all_reproduced':all(v['status']=='REPRODUCED' for v in C.values())};o=ROOT/a.out;o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
