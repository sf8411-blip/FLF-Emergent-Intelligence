from __future__ import annotations
import numpy as np
from .v419_dsl import evaluate, mdl_cost
from .v419_learner import propose_operator, random_proposal
from .v419_v418_adapter import fit_v418_l0
from .v419_worlds import make_world, reference_expr

PREREG_SHA256=None
PREREG_BLOB_SHA=None
DOWNSTREAM_BUDGET=48  # v4.18.5 default evolution budget; equal for B0/B1/L1/B2/B3.

def set_integrity(prereg_sha256:str, prereg_blob_sha:str):
    global PREREG_SHA256, PREREG_BLOB_SHA
    PREREG_SHA256=prereg_sha256; PREREG_BLOB_SHA=prereg_blob_sha


def _augment(env,u):
    z=dict(env)
    for j in range(u.shape[1]):z[f'u{j}']=u[:,j]
    return z

def _split(env,u,y):
    env=_augment(env,u);n=len(y);a=int(.58*n);b=int(.78*n)
    sub=lambda lo,hi:{k:v[lo:hi] for k,v in env.items()}
    return (sub(0,a),u[:a],y[:a]),(sub(a,b),u[a:b],y[a:b]),(sub(b,n),u[b:],y[b:])

def _fit(tr,val,te,extra,seed):
    return fit_v418_l0(tr[0],tr[1],tr[2],val[0],val[1],val[2],te[0],te[1],te[2],extra_exprs=extra,seed=seed,evaluation_budget=DOWNSTREAM_BUDGET)

def _canonical_signature(expr):
    r=np.random.default_rng(99173);n=160
    env={f'x{i}':r.normal(size=n) for i in range(3)};env.update({f'u{i}':r.normal(size=n) for i in range(2)})
    z=np.asarray(evaluate(expr,env),float);z=(z-z.mean())/(z.std()+1e-9);return z[:128].tolist()

def _candidate_accept(b0,cand,expr):
    mdl_gain=b0.description_length-cand.description_length-mdl_cost(expr)
    return cand.validation_rmse<=.92*b0.validation_rmse and mdl_gain>0,mdl_gain

def run_one(family:str,seed:int,budget:int,search_seed:int,tier:str,frozen:bool,control_seed:int,code_sha:str='UNKNOWN'):
    if not PREREG_SHA256:raise RuntimeError('set_integrity must be called before evaluation')
    env,u,y=make_world(family,seed,700,0);tr,val,te=_split(env,u,y)
    b0=_fit(tr,val,te,[],control_seed);b1=_fit(tr,val,te,[reference_expr(family)],control_seed)

    prop=propose_operator(tr[0],b0.train_residual,tr[1],budget,search_seed)
    accepted=False;expr=None;mdl_gain=0.0
    if prop.expr is not None:
        cand=_fit(tr,val,te,[prop.expr],control_seed);accepted,mdl_gain=_candidate_accept(b0,cand,prop.expr)
        if accepted:expr=prop.expr;l1=cand
        else:l1=b0
    else:l1=b0

    rnd_prop=random_proposal(tr[0],b0.train_residual,budget,search_seed+111);rnd=b0
    if rnd_prop.expr is not None:
        z=_fit(tr,val,te,[rnd_prop.expr],control_seed);ok,_=_candidate_accept(b0,z,rnd_prop.expr)
        if ok:rnd=z

    sh_prop=propose_operator(tr[0],b0.train_residual,tr[1],budget,search_seed+222,shuffled_residual=True);sh=b0
    if sh_prop.expr is not None:
        z=_fit(tr,val,te,[sh_prop.expr],control_seed);ok,_=_candidate_accept(b0,z,sh_prop.expr)
        if ok:sh=z
    dis_prop=propose_operator(tr[0],b0.train_residual,tr[1],budget,search_seed+333,trigger_enabled=False)

    if accepted:
        env2,u2,y2=make_world(family,seed+700000,700,1);tr2,val2,te2=_split(env2,u2,y2)
        tb0=_fit(tr2,val2,te2,[],control_seed+1);tb1=_fit(tr2,val2,te2,[reference_expr(family)],control_seed+1);tl1=_fit(tr2,val2,te2,[expr],control_seed+1)
        transfer_gap=(tb0.test_rmse-tl1.test_rmse)/max(tb0.test_rmse-tb1.test_rmse,1e-12)
    else:transfer_gap=float('nan')

    b0e=b0.test_rmse;b1e=b1.test_rmse;l1e=l1.test_rmse
    gap=lambda z:(b0e-z)/max(b0e-b1e,1e-12)
    return {
      'preregistration_sha256':PREREG_SHA256,'preregistration_git_blob_sha':PREREG_BLOB_SHA,'code_commit_sha':code_sha,
      'tier':tier,'frozen_test':frozen,'family':family,'seed':seed,'search_seed':search_seed,'control_seed':control_seed,
      'candidate_evaluations':prop.evaluations,'random_candidate_evaluations':rnd_prop.evaluations,
      'downstream_model_evaluations_per_arm':DOWNSTREAM_BUDGET,'rng_state_ref':f'numpy-pcg64:{search_seed}',
      'accepted':accepted,'operator':expr.text() if expr else None,'operator_depth':expr.depth() if expr else None,'operator_nodes':expr.nodes() if expr else None,
      'functional_signature':_canonical_signature(expr) if expr else None,'trigger_score':prop.trigger_score,
      'b0_error':b0e,'b1_error':b1e,'l1_error':l1e,'gap_closed':gap(l1e),
      'random_error':rnd.test_rmse,'random_gap_closed':gap(rnd.test_rmse),'shuffled_trigger_error':sh.test_rmse,'shuffled_gap_closed':gap(sh.test_rmse),
      'disabled_accepted':dis_prop.expr is not None,'transfer_gap_closed':transfer_gap,'ablation_damage':(b0e-l1e)/max(l1e,1e-12),
      'mdl_improvement':mdl_gain,'b0_residual_description_length':b0.description_length,
      'b0_feature_exprs':[e.text() for e in b0.exprs],'b0_genome':str(b0.genome),'b1_feature_exprs':[e.text() for e in b1.exprs]
    }

def run_control(control:str,seed:int,budget:int,search_seed:int,tier:str,frozen:bool,code_sha='UNKNOWN'):
    if not PREREG_SHA256:raise RuntimeError('set_integrity must be called before evaluation')
    family='null' if control=='null' else ('noise' if control=='noise' else 'actuator_shuffle')
    env,u,y=make_world(family,seed,700,0);tr,val,te=_split(env,u,y);b0=_fit(tr,val,te,[],search_seed+700)
    prop=propose_operator(tr[0],b0.train_residual,tr[1],budget,search_seed)
    accepted=False;expr=None;l1=b0
    if prop.expr is not None:
        cand=_fit(tr,val,te,[prop.expr],search_seed+700);accepted,_=_candidate_accept(b0,cand,prop.expr)
        if accepted:expr=prop.expr;l1=cand
    return {'preregistration_sha256':PREREG_SHA256,'preregistration_git_blob_sha':PREREG_BLOB_SHA,'code_commit_sha':code_sha,
            'tier':tier,'frozen_test':frozen,'family':control,'seed':seed,'search_seed':search_seed,'candidate_evaluations':prop.evaluations,
            'downstream_model_evaluations_per_arm':DOWNSTREAM_BUDGET,'rng_state_ref':f'numpy-pcg64:{search_seed}','accepted':accepted,
            'b0_error':b0.test_rmse,'l1_error':l1.test_rmse,'operator':expr.text() if expr else None,'trigger_score':prop.trigger_score}
