from __future__ import annotations
import numpy as np
from . import v420_dev as v
from . import v420_protocol as protocol
from . import v420_eval as ev

FAMILIES=('circular','rank','null','corrupt_circular','corrupt_rank','noise')
SEED_BASE={'circular':320000,'rank':321000,'null':322000,'corrupt_circular':323000,'corrupt_rank':324000,'noise':325000}
CANDIDATES=v.B2_CANDIDATES
MDL_LAMBDA=.002
TRIGGER_PERMUTATIONS=999
CHALLENGE_PERMUTATIONS=999
TRANSFER_ACTION_SCALE=1.6


def _gated_base(env,y,h):
    """Validated oracle instrument: B0 plus fixed validation-gated oracle coefficient."""
    b=v.b0_and_oracle(env,y,h)
    raw=b['oracle']; beta=float(raw['beta']); zz=np.asarray(raw['z'],float)
    opts=[]
    for scale in (0.,.25,.5,.75,1.):
        pred=b['pred']+(scale*beta)*zz
        opts.append((v.rmse(y[b['val']],pred[b['val']]),scale,pred))
    _,scale,pred=min(opts,key=lambda q:q[0])
    o=dict(raw);o['beta']=scale*beta;o['pred']=pred;o['val_rmse']=v.rmse(y[b['val']],pred[b['val']]);o['test_rmse']=v.rmse(y[b['te']],pred[b['te']]);o['validation_scale']=scale
    b['oracle']=o;b['b1_test_rmse']=o['test_rmse'];return b


def _circular_shift(y,seed):
    y=np.asarray(y,float);r=np.random.default_rng(int(seed)+818181);lo=max(1,len(y)//4);hi=max(lo+1,3*len(y)//4);shift=int(r.integers(lo,hi));return np.roll(y,shift),shift


def make_dress_world(family,seed):
    if family in ('corrupt_circular','corrupt_rank'):
        basefam='circular' if family=='corrupt_circular' else 'rank'
        env,y,h,meta=v.make_world(basefam,int(seed));ys,shift=_circular_shift(y,int(seed));return env,ys,np.zeros_like(h),{**meta,'corruption':'circular_shift_y','shift':shift}
    return v.make_world(family,int(seed))


def _source_meta(family,meta):
    return {k:meta[k] for k in ('period','window','quantile') if k in meta}


def _fresh_challenge_world(family,meta,seed):
    if family in ('circular','rank','null'):
        return protocol.make_trace_with_meta(family,int(seed),_source_meta(family,meta),action_scale=TRANSFER_ACTION_SCALE)
    if family in ('corrupt_circular','corrupt_rank'):
        basefam='circular' if family=='corrupt_circular' else 'rank'
        env,y,h,m=protocol.make_trace_with_meta(basefam,int(seed),_source_meta(basefam,meta),action_scale=TRANSFER_ACTION_SCALE)
        ys,shift=_circular_shift(y,int(seed)+19001);return env,ys,np.zeros_like(h),{**m,'corruption':'circular_shift_y','shift':shift}
    if family=='noise':return v.make_world('noise',int(seed),action_scale=TRANSFER_ACTION_SCALE)
    raise ValueError(family)


def _challenge_p(program,family,meta,seed,B=CHALLENGE_PERMUTATIONS):
    env,y,h,m=_fresh_challenge_world(family,meta,seed);base=_gated_base(env,y,h);z=v.eval_program(program,env);fit=v.extension_fit(base,y,z);actual=base['b0_val_rmse']-fit['val_rmse'];r=np.random.default_rng(int(seed)+404);null=[]
    for _ in range(int(B)):
        zp=v.block_permute(z,r,20);null.append(base['b0_val_rmse']-v.extension_fit(base,y,zp)['val_rmse'])
    pval=(1+sum(x>=actual for x in null))/(B+1)
    return float(pval),float(actual),m


def admit(program,z,fit,base,y,family,meta,seed):
    r2=v.novelty_r2(z,base['X'],base['tr']);gain=(base['b0_val_rmse']-fit['val_rmse'])/max(base['b0_val_rmse'],1e-12)
    d={'novelty_r2':float(r2),'validation_gain':float(gain),'challenge_p':1.0,'challenge_gain':0.0}
    if r2>v.NOVELTY_R2_MAX or gain<v.VALIDATION_GAIN_MIN:return False,d
    pv,cg,cm=_challenge_p(program,family,meta,seed);d.update({'challenge_p':pv,'challenge_gain':cg,'challenge_meta':cm})
    return bool(pv<=v.CHALLENGE_P_MAX),d


def _candidate_bank(env,base,seed,guided=False,shuffle=False,count=CANDIDATES):
    r=np.random.default_rng(int(seed));res=np.array(base['residual'],copy=True)
    if shuffle:res=v.block_permute(res,r,17)
    weights=None
    if guided:
        ks=sorted(env);w=np.asarray([abs(v.corr((env[k]-v.lag(env[k]))[base['tr']],res[base['tr']]))+.02 for k in ks]);weights=w/w.sum()
    out=[]
    for _ in range(int(count)):
        prog=v.sample_program(env,r)
        if guided and r.random()<.65:prog=v.replace(prog,a=str(r.choice(sorted(env),p=weights)))
        prog=protocol.tune_constants_continuous(prog,env,res,base['tr']);out.append((prog,v.eval_program(prog,env)))
    return out,res


def _select_guided(env,y,base,seed,shuffle=False):
    bank,res=_candidate_bank(env,base,seed,guided=True,shuffle=shuffle);sc=[]
    for prog,z in bank:
        fit=v.extension_fit(base,y,z);rc=abs(v.corr(z[base['tr']],res[base['tr']]));score=rc-MDL_LAMBDA*prog.mdl_cost();sc.append((score,prog,z,fit,rc))
    return max(sc,key=lambda q:q[0])


def _select_random_controls(env,y,base,seed):
    bank,res=_candidate_bank(env,base,seed,guided=False,shuffle=False);rows=[]
    for prog,z in bank:
        fit=v.extension_fit(base,y,z);rc=abs(v.corr(z[base['tr']],res[base['tr']]));rows.append((prog,z,fit,rc))
    b2a=max(rows,key=lambda q:q[3]);b2b=min(rows,key=lambda q:q[2]['val_rmse'])
    return b2a,b2b,[ev.program_signature(q[0]) for q in rows]


def _eval_arm(selected,base,y,family,meta,challenge_seed):
    if selected is None:return {'accepted':False,'program':None,'test_rmse':float(base['b0_test_rmse']),'novelty_r2':None,'validation_gain':None,'challenge_p':None,'mdl_cost':None,'constants':None}
    if len(selected)==5:_,prog,z,fit,_=selected
    else:prog,z,fit,_=selected
    ok,d=admit(prog,z,fit,base,y,family,meta,challenge_seed)
    return {'accepted':bool(ok),'program':prog,'fit':fit,'z_raw':z,'test_rmse':float(fit['test_rmse'] if ok else base['b0_test_rmse']),'novelty_r2':d['novelty_r2'],'validation_gain':d['validation_gain'],'challenge_p':d['challenge_p'],'challenge_gain':d.get('challenge_gain',0.0),'mdl_cost':float(prog.mdl_cost()),'constants':int(prog.constants()),'text':prog.text()}


def _transfer(program,family,meta,seed):
    env,y,h,m=protocol.make_trace_with_meta(family,int(seed),_source_meta(family,meta),action_scale=TRANSFER_ACTION_SCALE);base=_gated_base(env,y,h);z=v.eval_program(program,env);fit=v.extension_fit(base,y,z)
    return {'b0':float(base['b0_test_rmse']),'b1':float(base['b1_test_rmse']),'l1':float(fit['test_rmse']),'meta':m}


def run_seed(family,index):
    if family not in FAMILIES:raise ValueError(family)
    i=int(index);seed=SEED_BASE[family]+i;env,y,h,meta=make_dress_world(family,seed);base=_gated_base(env,y,h);actions=np.column_stack([env['u0'],env['u1']]);obs,thr,opened=v.calibrated_trigger(base['residual'][base['tr']],actions[base['tr']],seed+101,B=TRIGGER_PERMUTATIONS)
    validation_oracle_gain=(base['b0_val_rmse']-base['oracle']['val_rmse'])/max(base['b0_val_rmse'],1e-12)
    guided_sel=_select_guided(env,y,base,seed+1001,shuffle=False) if opened else None
    guided=_eval_arm(guided_sel,base,y,family,meta,seed+700001)
    b2a_sel,b2b_sel,bank_sig=_select_random_controls(env,y,base,seed+2001)
    b2a=_eval_arm(b2a_sel,base,y,family,meta,seed+710001);b2b=_eval_arm(b2b_sel,base,y,family,meta,seed+720001)
    b3_sel=_select_guided(env,y,base,seed+3001,shuffle=True)  # R8: always runs even if trigger is closed.
    b3=_eval_arm(b3_sel,base,y,family,meta,seed+730001)
    row={'family':family,'index':i,'seed':seed,'trigger_observed':float(obs),'trigger_threshold':float(thr),'trigger_open':bool(opened),'validation_oracle_gain':float(validation_oracle_gain),'b0_test_rmse':float(base['b0_test_rmse']),'b1_test_rmse':float(base['b1_test_rmse']),'oracle_den':float(base['b0_test_rmse']-base['b1_test_rmse']),'b3_ran':True,'random_bank_signature':[list(x) for x in bank_sig]}
    for name,arm in (('guided',guided),('b2a',b2a),('b2b',b2b),('b3',b3)):
        row[f'{name}_accepted']=bool(arm['accepted']);row[f'{name}_test_rmse']=float(arm['test_rmse']);row[f'{name}_num']=float(base['b0_test_rmse']-arm['test_rmse']);row[f'{name}_program']=arm.get('text');row[f'{name}_novelty_r2']=arm.get('novelty_r2');row[f'{name}_validation_gain']=arm.get('validation_gain');row[f'{name}_challenge_p']=arm.get('challenge_p');row[f'{name}_mdl_cost']=arm.get('mdl_cost');row[f'{name}_constants']=arm.get('constants')
    row['eligible']=bool(family in ('circular','rank') and validation_oracle_gain>=.05)
    row['functional_similarity']=None;row['predictive_substitutability']=None;row['ablation_rmse']=None;row['ablation_num']=None;row['transfer_b0']=None;row['transfer_b1']=None;row['transfer_l1']=None;row['transfer_num']=None;row['transfer_den']=None
    if family in ('circular','rank') and guided['accepted']:
        te=base['te'];zraw=np.asarray(guided['z_raw']);row['functional_similarity']=abs(v.corr(zraw[te],h[te]));row['predictive_substitutability']=ev.predictive_substitutability(guided['fit']['beta']*guided['fit']['z'][te],base['oracle']['beta']*base['oracle']['z'][te]);abl=ev.in_model_block_ablation(y,base['pred'],guided['fit']['beta'],guided['fit']['z'],te,seed+800001,block=20);row['ablation_rmse']=float(abl);row['ablation_num']=float(abl-guided['test_rmse']);trf=_transfer(guided['program'],family,meta,seed+900001);row['transfer_b0']=trf['b0'];row['transfer_b1']=trf['b1'];row['transfer_l1']=trf['l1'];row['transfer_num']=trf['b0']-trf['l1'];row['transfer_den']=trf['b0']-trf['b1']
    elif family in ('circular','rank'):
        # A rejected/no-proposal method has zero transfer gain by definition; use the same fresh trace for the oracle denominator.
        env2,y2,h2,m2=protocol.make_trace_with_meta(family,seed+900001,_source_meta(family,meta),action_scale=TRANSFER_ACTION_SCALE);bb=_gated_base(env2,y2,h2);row['transfer_b0']=float(bb['b0_test_rmse']);row['transfer_b1']=float(bb['b1_test_rmse']);row['transfer_l1']=float(bb['b0_test_rmse']);row['transfer_num']=0.0;row['transfer_den']=float(bb['b0_test_rmse']-bb['b1_test_rmse'])
    return row
