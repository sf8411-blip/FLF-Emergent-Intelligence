#!/usr/bin/env python3
from __future__ import annotations
import argparse, inspect, json, math, pathlib, re, statistics, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flf_ecf.v419_worlds import make_world, hidden_feature
from flf_ecf.v419_dsl import Expr, l0_expressions, reference_expr, mdl_cost, random_proposal
from flf_ecf.v419_learner import trigger_components, TRIGGER_THRESHOLD
from flf_ecf.v419_v418_adapter import fit_b0_v418


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3:return 0.0
    a=a[m];b=b[m]
    if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
    return float(np.corrcoef(a,b)[0,1])

def top_split_args(s):
    out=[];st=0;start=0
    for i,ch in enumerate(s):
        if ch=='(':st+=1
        elif ch==')':st-=1
        elif ch==',' and st==0:out.append(s[start:i].strip());start=i+1
    out.append(s[start:].strip())
    return out

def parse_expr(text):
    text=text.strip()
    if '(' not in text:return Expr.var(text)
    p=text.find('('); head=text[:p]; inner=text[p+1:-1]
    if '[' in head:
        op=head[:head.index('[')]; raw=head[head.index('[')+1:-1]
        try:param=int(raw)
        except ValueError:param=float(raw)
    else:op=head;param=None
    args=tuple(parse_expr(x) for x in top_split_args(inner))
    return Expr(op,args,param)

def augment(world):
    env={k:np.asarray(v,float) for k,v in world['env'].items()}
    u=np.asarray(world['actions'],float)
    for j in range(u.shape[1]):env[f'u{j}']=u[:,j]
    return env

def r2_in_l0(expr,env,mask=None):
    names=sorted(env)
    cols=[]
    for e in l0_expressions(names):
        try:cols.append(np.asarray(e.eval(env),float))
        except Exception:pass
    y=np.asarray(expr.eval(env),float)
    X=np.column_stack(cols)
    if mask is None:mask=np.ones(len(y),bool)
    m=np.asarray(mask,bool)&np.isfinite(y)&np.all(np.isfinite(X),axis=1)
    X=X[m];y=y[m]
    X=np.column_stack([np.ones(len(X)),X])
    beta=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@beta
    sst=float(np.sum((y-y.mean())**2));sse=float(np.sum((y-pred)**2))
    return 1.0-sse/max(sst,1e-12)

def pairwise_median(vs):
    vals=[]
    for i in range(len(vs)):
        for j in range(i+1,len(vs)):
            vals.append(abs(corr(vs[i],vs[j])))
    return float(np.median(vals)) if vals else 0.0

def approx(v,target,tol):return abs(float(v)-float(target))<=tol

def load_rows(path):return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',default='results/v4.19-t2-raw.jsonl');ap.add_argument('--out',default='results/v4.20-audit-v419.json');ap.add_argument('--noise-sims',type=int,default=10000);a=ap.parse_args()
    rows=load_rows(ROOT/a.raw)
    by={f:[r for r in rows if r['family']==f] for f in sorted({r['family'] for r in rows})}
    claims={};diag={}

    # (a) mechanism retained in actuator-shuffle; operator relation + L0-span audit.
    ccor=[];acor=[];opcor=[];opr2=[]
    for r in by.get('circular',[]):
        w=make_world('circular',int(r['seed']));ccor.append(abs(corr(w['y'],hidden_feature('circular',w['env']))))
    for r in by.get('actuator_shuffle',[]):
        w=make_world('actuator_shuffle',int(r['seed']));h=hidden_feature('circular',w['env']);acor.append(abs(corr(w['y'],h)))
        if r.get('accepted') and r.get('operator'):
            e=parse_expr(r['operator']);env=augment(w);opcor.append(abs(corr(e.eval(env),h)));opr2.append(r2_in_l0(e,env))
    src_world=inspect.getsource(make_world)
    a_ok=(len(opcor)==23 and approx(np.mean(ccor),.887,.04) and approx(np.mean(acor),.878,.04) and approx(np.median(opcor),.86,.10) and approx(np.median(opr2),.09,.12) and '0.08' in src_world and 'u[::-1,0]' in src_world.replace(' ',''))
    diag['a']={'circular_y_hidden_corr_mean':float(np.mean(ccor)),'actuator_shuffle_y_hidden_corr_mean':float(np.mean(acor)),'accepted':len(opcor),'operator_hidden_corr_median':float(np.median(opcor)) if opcor else None,'operator_l0_r2_median':float(np.median(opr2)) if opr2 else None}
    claims['a']={'status':'REPRODUCED' if a_ok else 'NOT REPRODUCED','diagnostics':diag['a']}

    # (b) accepted Null operators are L0-span; trigger large; selector has 3 slots; no novelty gate.
    nr2=[];nops=[]
    for r in by.get('null',[]):
        if r.get('accepted') and r.get('operator'):
            w=make_world('null',int(r['seed']));env=augment(w);e=parse_expr(r['operator']);nr2.append(r2_in_l0(e,env));nops.append(r['operator'])
    null_min=min(float(r.get('trigger_score',0)) for r in by.get('null',[]))
    adapter_src=inspect.getsource(fit_b0_v418)
    import flf_ecf.v419_learner as learner
    learner_src=inspect.getsource(learner.evaluate_residual_episode)
    b_ok=(len(nr2)==14 and np.median(nr2)>=.99 and null_min>=9.0 and ('[:3]' in adapter_src or 'max_features=3' in adapter_src or '3' in adapter_src) and 'novel' not in learner_src.lower())
    diag['b']={'accepted':len(nr2),'l0_r2_median':float(np.median(nr2)) if nr2 else None,'trigger_min':null_min,'example':nops[0] if nops else None}
    claims['b']={'status':'REPRODUCED' if b_ok else 'NOT REPRODUCED','diagnostics':diag['b']}

    # (c) threshold calibration vs iid simulation + raw noise opens.
    noise_open=sum(float(r.get('trigger_score',0))>=TRIGGER_THRESHOLD for r in by.get('noise',[]))
    rng=np.random.default_rng(44001);opens=0;n=406
    for _ in range(a.noise_sims):
        residual=rng.normal(size=n);actions=rng.normal(size=(n,2));score=float(np.sum(trigger_components(residual,actions)))
        opens += score>=TRIGGER_THRESHOLD
    p_open=opens/a.noise_sims
    c_ok=(noise_open==12 and approx(p_open,.51,.08) and approx(TRIGGER_THRESHOLD,2.5,1e-12))
    diag['c']={'threshold':TRIGGER_THRESHOLD,'simulated_iid_open_rate':p_open,'noise_open_count':noise_open,'noise_n':len(by.get('noise',[]))}
    claims['c']={'status':'REPRODUCED' if c_ok else 'NOT REPRODUCED','diagnostics':diag['c']}

    # (d) old iid signatures vs world-trace semantics + hidden feature corr.
    circacc=[r for r in by.get('circular',[]) if r.get('accepted') and r.get('operator')]
    iid=[r['functional_signature'] for r in circacc if r.get('functional_signature')]
    iidmed=pairwise_median(iid)
    # Evaluate every accepted operator on common world traces, standardize each per trace, concatenate.
    world_vectors=[];hidden_corrs=[]
    common_seeds=[int(r['seed']) for r in by.get('circular',[])[:8]]
    for r in circacc:
        e=parse_expr(r['operator']);pieces=[]
        for s in common_seeds:
            w=make_world('circular',s);env=augment(w);z=np.asarray(e.eval(env),float);z=(z-np.nanmean(z))/(np.nanstd(z)+1e-12);pieces.append(np.nan_to_num(z))
        world_vectors.append(np.concatenate(pieces))
        w=make_world('circular',int(r['seed']));hidden_corrs.append(abs(corr(e.eval(augment(w)),hidden_feature('circular',w['env']))))
    wmed=pairwise_median(world_vectors);hmed=float(np.median(hidden_corrs)) if hidden_corrs else 0.0
    d_ok=(len(circacc)==22 and approx(iidmed,.164,.05) and approx(wmed,.877,.12) and approx(hmed,.963,.08))
    diag['d']={'accepted':len(circacc),'iid_signature_pairwise_median':iidmed,'world_trace_pairwise_median':wmed,'hidden_feature_corr_median':hmed}
    claims['d']={'status':'REPRODUCED' if d_ok else 'NOT REPRODUCED','diagnostics':diag['d']}

    # (e) B2 selection path; shuffled exact zeros; G3 algebraic dependence.
    dsl_src=inspect.getsource(random_proposal)
    random_g=[float(r.get('random_gap_closed',0)) for r in by.get('circular',[])]
    nonnull=by.get('circular',[])+by.get('rank_interaction',[])
    shuf_zero=all(abs(float(r.get('shuffled_gap_closed',0)))<1e-12 for r in nonnull)
    g3_exact=all(abs(float(r.get('ablation_damage',0)) - ((float(r['b0_error'])-float(r.get('l1_error',r['b0_error'])))/max(float(r.get('l1_error',r['b0_error'])),1e-12)))<1e-9 for r in nonnull)
    e_ok=('residual' in dsl_src and 'corr' in dsl_src and 'pool' in dsl_src and np.mean(random_g)>.75 and all(x>0 for x in random_g) and shuf_zero and g3_exact)
    diag['e']={'random_gap_mean':float(np.mean(random_g)),'random_positive_count':sum(x>0 for x in random_g),'circular_n':len(random_g),'shuffled_zero_48':shuf_zero,'g3_transform_exact':g3_exact}
    claims['e']={'status':'REPRODUCED' if e_ok else 'NOT REPRODUCED','diagnostics':diag['e']}

    # (f) world constants and MDL constant charge absence; cumsum frequency.
    texts=[r['operator'] for r in circacc]
    mod1=sum('modulo[1.0]' in t for t in texts);cumsum=sum('cumsum' in t for t in texts)
    rank_ref=reference_expr('rank_interaction').text();mdl_src=inspect.getsource(mdl_cost)
    f_ok=(mod1==len(texts)==22 and all(s in rank_ref for s in ('rank01[9]','threshold_gt[0.75]','lag[3]')) and ('learned_constants=0' in mdl_src.replace(' ','')) and cumsum==0)
    diag['f']={'circular_accepted':len(texts),'modulo_1_count':mod1,'cumsum_count':cumsum,'rank_reference':rank_ref,'mdl_source_has_zero_constants':'learned_constants=0' in mdl_src.replace(' ','')}
    claims['f']={'status':'REPRODUCED' if f_ok else 'NOT REPRODUCED','diagnostics':diag['f']}

    # (g) rank oracle instability, influential ratio, pooled ratio.
    rank=by.get('rank_interaction',[]);og=[(r['b0_error']-r['b1_error'])/max(r['b0_error'],1e-12) for r in rank]
    gaps=[float(r['gap_closed']) for r in rank];imax=int(np.argmax(gaps)) if gaps else -1
    pooled=(sum(r['b0_error']-r.get('l1_error',r['b0_error']) for r in rank)/max(sum(r['b0_error']-r['b1_error'] for r in rank),1e-12)) if rank else 0
    g_ok=(sum(x<0 for x in og)==4 and approx(np.mean(gaps),.34,.03) and imax>=0 and approx(og[imax],.008,.015) and approx(gaps[imax],7.77,.8) and approx(pooled,.048,.03))
    diag['g']={'negative_oracle_gain_count':sum(x<0 for x in og),'mean_gap':float(np.mean(gaps)) if gaps else None,'max_gap_seed':rank[imax]['seed'] if imax>=0 else None,'max_gap':gaps[imax] if imax>=0 else None,'max_gap_oracle_gain':og[imax] if imax>=0 else None,'pooled_ratio':float(pooled)}
    claims['g']={'status':'REPRODUCED' if g_ok else 'NOT REPRODUCED','diagnostics':diag['g']}

    # (h) historical T1 claim requires exact archived pilot bytes. If absent, do not infer from later summaries.
    t1=ROOT/'results/v4.19-t1-summary.json'
    if t1.exists():
        tj=json.loads(t1.read_text());fam=tj.get('families',{});null=fam.get('null',{}).get('acceptance_rate');rk=fam.get('rank_interaction',{}).get('acceptance_rate');rand=fam.get('rank_interaction',{}).get('mean_random_gap_closed');guid=fam.get('rank_interaction',{}).get('mean_gap_closed');h_ok=(approx(null,.50,.02) and approx(rk,.20,.02) and rand is not None and guid is not None and rand>guid);diag['h']={'archived_summary':str(t1),'null_acceptance':null,'rank_acceptance':rk,'rank_random':rand,'rank_guided':guid}
    else:
        h_ok=False;diag['h']={'reason':'Exact committed T1 summary/raw artifact is not present in the closed branch; historical STATE warning alone is insufficient for independent numerical reproduction.'}
    claims['h']={'status':'REPRODUCED' if h_ok else 'NOT REPRODUCED','diagnostics':diag['h']}

    out={'source_raw':a.raw,'n_rows':len(rows),'claims':claims,'all_reproduced':all(v['status']=='REPRODUCED' for v in claims.values())}
    path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
