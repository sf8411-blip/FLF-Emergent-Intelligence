from __future__ import annotations
import numpy as np
from . import v420_dev as v

def pooled_ratio(rows,num_key='num',den_key='den'):
    den=sum(float(r[den_key]) for r in rows)
    return sum(float(r[num_key]) for r in rows)/max(den,1e-12)

def bootstrap_pooled(rows,num_key='num',den_key='den',B=4000,seed=2020):
    rr=np.random.default_rng(seed);n=len(rows);vals=[]
    for _ in range(B):
        ix=rr.integers(0,n,n);sample=[rows[int(i)] for i in ix];vals.append(pooled_ratio(sample,num_key,den_key))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def in_model_block_ablation(y,base_pred,beta,z,test_idx,seed,block=20):
    """Permute only the fitted extension feature on test; never refit any parameter."""
    y=np.asarray(y,float);base_pred=np.asarray(base_pred,float);z=np.asarray(z,float);idx=np.asarray(test_idx,int)
    rr=np.random.default_rng(seed);zp=np.array(z,copy=True);zp[idx]=v.block_permute(z[idx],rr,block)
    pred=base_pred+float(beta)*zp
    return v.rmse(y[idx],pred[idx])

def pairwise_abs_corr(signatures):
    vals=[]
    for i in range(len(signatures)):
        for j in range(i+1,len(signatures)):
            vals.append(abs(v.corr(signatures[i],signatures[j])))
    return float(np.median(vals)) if vals else 0.0

def predictive_substitutability(contrib_a,contrib_b):
    return abs(v.corr(np.asarray(contrib_a,float),np.asarray(contrib_b,float)))

def program_signature(p):
    """Stable structural identity that never formats absent constants."""
    return (p.kind,p.a,p.b,None if p.p is None else float(p.p),None if p.q is None else float(p.q))

def control_candidates(env,base,seed,trigger_open=False,count=v.B2_CANDIDATES):
    """R8 controls: B2a/B2b share the same random bank; B3 always searches."""
    bank,res=v.candidate_bank(env,base,seed,count=count,guided=False,shuffle=False)
    def choose(mode):
        scored=[]
        for p,z in bank:
            m=v.extension_fit(base,np.asarray(base['_y']),z)
            rc=abs(v.corr(z[base['tr']],res[base['tr']]))
            scored.append((-rc if mode=='B2a' else m['val_rmse'],p,z,m))
        return min(scored,key=lambda q:q[0])
    b2a=choose('B2a');b2b=choose('B2b')
    # Deliberately ignore trigger_open: B3 is an equal-budget search control.
    b3=v.select_candidate(env,np.asarray(base['_y']),base,seed+1,'shuffled',count=count)
    return {'B2a':b2a,'B2b':b2b,'B3':b3,'b3_ran':True,'trigger_open_input':bool(trigger_open),'bank_signature':[program_signature(p) for p,_ in bank]}
