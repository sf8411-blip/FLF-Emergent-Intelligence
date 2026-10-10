from __future__ import annotations
import numpy as np
from .v419_dsl import l0_expressions
from .v419_learner import linear_fit, apply_fit, synthesize_operator, random_composition_control
from .v419_worlds import make_world, reference_expr

PREREG_SHA256='c870ac245b5c4ed1d70c32e1ba97dd87e48106d7fbb6807ed167bca9d2bf4e19'
PREREG_BLOB_SHA='d8da1bc2c2ab439e0dc65982642c5788fb7c2d02'


def _split(env,u,y):
    n=len(y);a=int(.58*n);b=int(.78*n)
    sub=lambda lo,hi:{k:v[lo:hi] for k,v in env.items()}
    return (sub(0,a),u[:a],y[:a]),(sub(a,b),u[a:b],y[a:b]),(sub(b,n),u[b:],y[b:])

def _oracle_error(family,tr,te):
    trE,trU,trY=tr;teE,teU,teY=te
    exprs=l0_expressions(sorted(trE))+[reference_expr(family)]
    f=linear_fit(exprs,trE,trY);err,_=apply_fit(exprs,f,teE,teY);return err

def _b0_error(tr,te):
    trE,trU,trY=tr;teE,teU,teY=te;exprs=l0_expressions(sorted(trE));f=linear_fit(exprs,trE,trY);err,_=apply_fit(exprs,f,teE,teY);return err,f

def _with_expr_error(expr,tr,te):
    trE,trU,trY=tr;teE,teU,teY=te;exprs=l0_expressions(sorted(trE))+[expr];f=linear_fit(exprs,trE,trY);err,_=apply_fit(exprs,f,teE,teY);return err,f

def run_one(family:str,seed:int,budget:int,search_seed:int,tier:str,frozen:bool,control_seed:int,code_sha:str='UNKNOWN'):
    env,u,y=make_world(family,seed,700,0);tr,val,te=_split(env,u,y)
    b0,bfit=_b0_error(tr,te)
    b1=_oracle_error(family,tr,te) if family in ('circular','rank_interaction') else b0
    sr=synthesize_operator(tr[0],tr[2],val[0],val[2],tr[1],budget,search_seed)
    if sr.accepted:
        l1,_=_with_expr_error(sr.expr,tr,te); ablated=b0
        env2,u2,y2=make_world(family,seed+700000,700,1);tr2,val2,te2=_split(env2,u2,y2)
        transfer_b0,_=_b0_error(tr2,te2);transfer_b1=_oracle_error(family,tr2,te2)
        transfer_l1,_=_with_expr_error(sr.expr,tr2,te2)
    else:
        l1=b0;ablated=b0;transfer_b0=transfer_b1=transfer_l1=float('nan')
    rnd_val,rnd_expr,rnd_evals=random_composition_control(tr[0],tr[2],val[0],val[2],tr[1],budget,search_seed+111)
    rnd_err=b0 if rnd_expr is None else _with_expr_error(rnd_expr,tr,te)[0]
    sh=synthesize_operator(tr[0],tr[2],val[0],val[2],tr[1],budget,search_seed+222,shuffled_residual=True)
    sh_err=b0 if not sh.accepted else _with_expr_error(sh.expr,tr,te)[0]
    dis=synthesize_operator(tr[0],tr[2],val[0],val[2],tr[1],budget,search_seed+333,trigger_enabled=False)
    gap=lambda z:(b0-z)/max(b0-b1,1e-12) if family in ('circular','rank_interaction') else 0.0
    transfer_gap=(transfer_b0-transfer_l1)/max(transfer_b0-transfer_b1,1e-12) if sr.accepted and family in ('circular','rank_interaction') else float('nan')
    return {
      'preregistration_sha256':PREREG_SHA256,'preregistration_git_blob_sha':PREREG_BLOB_SHA,'code_commit_sha':code_sha,
      'tier':tier,'frozen_test':frozen,'family':family,'seed':seed,'search_seed':search_seed,'control_seed':control_seed,
      'candidate_evaluations':sr.evaluations,'random_candidate_evaluations':rnd_evals,'rng_state_ref':f'numpy-pcg64:{search_seed}',
      'accepted':sr.accepted,'operator':sr.expr.text() if sr.accepted else None,'operator_depth':sr.expr.depth() if sr.accepted else None,
      'operator_nodes':sr.expr.nodes() if sr.accepted else None,'functional_signature':sr.trace_signature,
      'trigger_score':sr.trigger_score,'b0_error':b0,'b1_error':b1,'l1_error':l1,'gap_closed':gap(l1),
      'random_error':rnd_err,'random_gap_closed':gap(rnd_err),'shuffled_trigger_error':sh_err,'shuffled_gap_closed':gap(sh_err),
      'disabled_accepted':dis.accepted,'transfer_gap_closed':transfer_gap,'ablation_damage':(ablated-l1)/max(l1,1e-12),
      'mdl_improvement':sr.mdl_improvement,'b0_residual_description_length':bfit.dl,
    }

def run_control(control:str,seed:int,budget:int,search_seed:int,tier:str,frozen:bool,code_sha='UNKNOWN'):
    family='null' if control=='null' else ('noise' if control=='noise' else 'actuator_shuffle')
    env,u,y=make_world(family,seed,700,0);tr,val,te=_split(env,u,y)
    b0,_=_b0_error(tr,te)
    sr=synthesize_operator(tr[0],tr[2],val[0],val[2],tr[1],budget,search_seed)
    l1=b0 if not sr.accepted else _with_expr_error(sr.expr,tr,te)[0]
    return {'preregistration_sha256':PREREG_SHA256,'preregistration_git_blob_sha':PREREG_BLOB_SHA,'code_commit_sha':code_sha,
            'tier':tier,'frozen_test':frozen,'family':control,'seed':seed,'search_seed':search_seed,'candidate_evaluations':sr.evaluations,
            'rng_state_ref':f'numpy-pcg64:{search_seed}','accepted':sr.accepted,'b0_error':b0,'l1_error':l1,'operator':sr.expr.text() if sr.accepted else None}
