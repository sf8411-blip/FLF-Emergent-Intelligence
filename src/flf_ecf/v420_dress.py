from __future__ import annotations
import numpy as np
from . import v420_repair as r
from . import v420_dev as old
from . import v420_eval as ev

FAMILIES=('circular','rank','null','corrupt_circular','corrupt_rank','noise','spurious')
SEED_BASE={'circular':320000,'rank':321000,'null':322000,'corrupt_circular':323000,'corrupt_rank':324000,'noise':325000,'spurious':326000}
DRESS_RANGES=tuple(range(v,v+32) for v in SEED_BASE.values())
CHALLENGE_PERMUTATIONS=999
TRANSFER_ACTION_SCALE=1.6

def _is_dress_seed(seed):return any(int(seed) in q for q in DRESS_RANGES)

def _transfer(expr,family,meta,seed):
    env,y,h,m=r.make_world(family,int(seed),action_scale=TRANSFER_ACTION_SCALE,meta=meta);b=r.b0_and_oracle(env,y,h);z=r.evaluate(expr,env);fit=r.extension_fit(b,y,z)
    return {'b0':float(b['b0_test_rmse']),'b1':float(b['b1_test_rmse']),'l1':float(fit['test_rmse']),'meta':m}

def run_seed(family,index=None,*,seed_override=None,challenge_B=CHALLENGE_PERMUTATIONS,search_budget=None):
    """Single authoritative v4.20 execution path.

    Dress uses ``index`` and the preregistered family base. Smoke/validation may
    use ``seed_override`` only when the seed is outside every dress range.
    """
    if family not in FAMILIES:raise ValueError(family)
    if seed_override is None:
        if index is None:raise ValueError('dress run requires index')
        i=int(index);seed=SEED_BASE[family]+i
        if not 0<=i<32:raise ValueError('dress index must be 0..31')
    else:
        seed=int(seed_override);i=-1
        if _is_dress_seed(seed):raise ValueError('seed_override must never use a dress seed')
    budget=r.SEARCH_BUDGET if search_budget is None else int(search_budget)
    core=r.run_seed_core(family,seed,challenge_B=int(challenge_B),search_budget=budget)
    b=core['base'];y=np.asarray(core['y']);h=np.asarray(core['h']);meta=core['meta']
    validation_oracle_gain=(b['b0_val_rmse']-b['oracle']['val_rmse'])/max(b['b0_val_rmse'],1e-12)
    row={'family':family,'index':i,'seed':seed,'trigger_observed':core['trigger_observed'],'trigger_threshold':core['trigger_threshold'],'trigger_open':core['trigger_open'],'trigger_is_gate':False,'validation_oracle_gain':float(validation_oracle_gain),'b0_test_rmse':float(b['b0_test_rmse']),'b1_test_rmse':float(b['b1_test_rmse']),'oracle_den':float(b['b0_test_rmse']-b['b1_test_rmse']),'b3_ran':True,'search_budget':budget,'search_counts':core['search_counts'],'b2_same_program':core['b2a']['signature']==core['b2b']['signature']}
    for name in ('guided','b2a','b2b','b3'):
        arm=core[name];row[f'{name}_accepted']=bool(arm['accepted']);row[f'{name}_test_rmse']=float(arm['test_rmse']);row[f'{name}_num']=float(b['b0_test_rmse']-arm['test_rmse']);row[f'{name}_program']=arm['program'];row[f'{name}_novelty_r2']=arm['novelty_r2'];row[f'{name}_validation_gain']=arm['validation_gain'];row[f'{name}_challenge_p']=arm['challenge_p'];row[f'{name}_mdl_cost']=float(arm['mdl_cost']);row[f'{name}_constants']=int(arm['constants'])
    row['random_bank_signature']=core['random_bank_signature']
    row['eligible']=bool(family in ('circular','rank') and validation_oracle_gain>=.05)
    row['functional_similarity']=None;row['predictive_substitutability']=None;row['ablation_rmse']=None;row['ablation_num']=None;row['transfer_b0']=None;row['transfer_b1']=None;row['transfer_l1']=None;row['transfer_num']=None;row['transfer_den']=None
    g=core['guided']
    if family in ('circular','rank') and g['accepted']:
        te=b['te'];zraw=np.asarray(g['z']);fit=g['fit'];row['functional_similarity']=abs(old.corr(zraw[te],h[te]));row['predictive_substitutability']=ev.predictive_substitutability(fit['beta']*fit['z'][te],b['oracle']['beta']*b['oracle']['z'][te]);abl=ev.in_model_block_ablation(y,b['pred'],fit['beta'],fit['z'],te,seed+800001,block=20);row['ablation_rmse']=float(abl);row['ablation_num']=float(abl-g['test_rmse']);trf=_transfer(g['expr'],family,meta,seed+900001);row['transfer_b0']=trf['b0'];row['transfer_b1']=trf['b1'];row['transfer_l1']=trf['l1'];row['transfer_num']=trf['b0']-trf['l1'];row['transfer_den']=trf['b0']-trf['b1']
    elif family in ('circular','rank'):
        env2,y2,h2,m2=r.make_world(family,seed+900001,action_scale=TRANSFER_ACTION_SCALE,meta=meta);bb=r.b0_and_oracle(env2,y2,h2);row['transfer_b0']=float(bb['b0_test_rmse']);row['transfer_b1']=float(bb['b1_test_rmse']);row['transfer_l1']=float(bb['b0_test_rmse']);row['transfer_num']=0.0;row['transfer_den']=float(bb['b0_test_rmse']-bb['b1_test_rmse'])
    return row
