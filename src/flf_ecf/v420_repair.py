from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from . import v420_dev as base

VALIDATION_GAIN_MIN=.05
NOVELTY_R2_MAX=.50
CHALLENGE_P_MAX=.01
MDL_CONSTANT_COST=.5
MDL_LAMBDA=.002
SEARCH_BUDGET=128  # iteration-0 R4 calibration value; identical for all four search arms.
MAX_DEPTH=6
MAX_NODES=11

PRIMITIVES=('lag','diff','cumsum','filter_mean','rank','threshold','modulo','product','interaction','negate','abs','add','subtract')
UNARY=('lag','diff','cumsum','filter_mean','rank','threshold','modulo','negate','abs')
BINARY=('product','interaction','add','subtract')
PARAM_OPS=('filter_mean','rank','threshold','modulo')
FORBIDDEN_TEMPLATE_OPS=('moddiff','rankprod','wrap_correct','rank_interaction')

@dataclass(frozen=True)
class Expr:
    op:str
    args:tuple['Expr',...]=()
    param:float|str|None=None
    def depth(self):return 0 if self.op=='var' else 1+max(a.depth() for a in self.args)
    def nodes(self):return 1+sum(a.nodes() for a in self.args)
    def constants(self):return int(self.op in PARAM_OPS)+sum(a.constants() for a in self.args)
    def mdl_cost(self):return float(self.nodes()+MDL_CONSTANT_COST*self.constants())
    def text(self):
        if self.op=='var':return str(self.param)
        p='' if self.param is None else f'[{float(self.param):.6g}]'
        return f'{self.op}{p}('+','.join(a.text() for a in self.args)+')'

def var(name):return Expr('var',(),str(name))

def expr_type(e):
    if e.op=='threshold':return 'bool'
    return 'scalar'

def _valid_unary(op,a):
    return expr_type(a) in ('scalar','bool') and not (op in ('rank','modulo') and expr_type(a)!='scalar')

def _valid_binary(op,a,b):
    if op=='interaction':return expr_type(a)=='scalar' and expr_type(b)=='bool'
    return expr_type(a) in ('scalar','bool') and expr_type(b) in ('scalar','bool')

def evaluate(e,env):
    if e.op=='var':return np.asarray(env[str(e.param)],float)
    xs=[evaluate(a,env) for a in e.args]
    if e.op=='lag':return base.lag(xs[0],1)
    if e.op=='diff':return xs[0]-base.lag(xs[0],1)
    if e.op=='cumsum':return np.cumsum(xs[0])
    if e.op=='filter_mean':return base.expmean(xs[0],float(e.param))
    if e.op=='rank':return base.weighted_rank(xs[0],float(e.param))
    if e.op=='threshold':return (xs[0]>float(e.param)).astype(float)
    if e.op=='modulo':return np.mod(xs[0],max(float(e.param),1e-6))
    if e.op=='product':return xs[0]*xs[1]
    if e.op=='interaction':return xs[0]*(xs[1]>.5)
    if e.op=='negate':return -xs[0]
    if e.op=='abs':return np.abs(xs[0])
    if e.op=='add':return xs[0]+xs[1]
    if e.op=='subtract':return xs[0]-xs[1]
    raise ValueError(e.op)

def _walk_param_paths(e,path=()):
    out=[]
    if e.op in PARAM_OPS:out.append(path)
    for i,a in enumerate(e.args):out.extend(_walk_param_paths(a,path+(i,)))
    return out

def _get(e,path):
    q=e
    for i in path:q=q.args[i]
    return q

def _set(e,path,new):
    if not path:return new
    i=path[0];aa=list(e.args);aa[i]=_set(aa[i],path[1:],new);return replace(e,args=tuple(aa))

def _bounds(node,arg_values=None):
    if node.op=='filter_mean':return (1.5,8.0)
    if node.op=='rank':return (4.0,16.0)
    if node.op=='modulo':return (.45,1.65)
    if node.op=='threshold':
        if arg_values is None:return (-1.2,1.2)
        a=np.asarray(arg_values,float);lo=float(np.quantile(a,.01));hi=float(np.quantile(a,.99))
        if not np.isfinite(lo+hi) or hi-lo<1e-4:return (-1.2,1.2)
        pad=.1*(hi-lo);return (lo-pad,hi+pad)
    raise ValueError(node.op)

def _search1d(fn,lo,hi,rounds=3,points=7):
    a,b=float(lo),float(hi);best=(a+b)/2
    for _ in range(rounds):
        xs=np.linspace(a,b,points);ys=np.asarray([fn(float(x)) for x in xs],float);j=int(np.argmax(ys));best=float(xs[j])
        if j==0:a,b=float(xs[0]),float(xs[1])
        elif j==len(xs)-1:a,b=float(xs[-2]),float(xs[-1])
        else:a,b=float(xs[j-1]),float(xs[j+1])
    return best

def tune_constants(e,env,target,idx,passes=2):
    idx=np.asarray(idx,int);tar=np.asarray(target,float);out=e
    paths=_walk_param_paths(out)
    for _ in range(passes if len(paths)>1 else 1):
        for path in paths:
            node=_get(out,path)
            argvals=evaluate(node.args[0],env)[idx] if node.op=='threshold' else None
            lo,hi=_bounds(node,argvals)
            def score(x):
                q=_set(out,path,replace(node,param=float(x)))
                z=evaluate(q,env)
                return abs(base.corr(z[idx],tar[idx]))
            x=_search1d(score,lo,hi);out=_set(out,path,replace(node,param=float(x)))
    return out

def _init_param(op,rng):
    if op=='filter_mean':return float(rng.uniform(1.5,8.0))
    if op=='rank':return float(rng.uniform(4.0,16.0))
    if op=='threshold':return float(rng.uniform(-.8,.8))
    if op=='modulo':return float(rng.uniform(.45,1.65))
    return None

def random_expr(variables,rng,max_depth=MAX_DEPTH,max_nodes=MAX_NODES,var_weights=None):
    variables=tuple(variables)
    def leaf():
        if var_weights is None:return var(variables[int(rng.integers(len(variables)))])
        return var(str(rng.choice(variables,p=var_weights)))
    def rec(depth):
        if depth<=0 or rng.random()<.16:return leaf()
        if rng.random()<.58:
            op=UNARY[int(rng.integers(len(UNARY)))];a=rec(depth-1)
            if not _valid_unary(op,a):return leaf()
            return Expr(op,(a,),_init_param(op,rng))
        op=BINARY[int(rng.integers(len(BINARY)))];a=rec(depth-1);b=rec(max(0,depth-1-int(rng.random()<.4)))
        if not _valid_binary(op,a,b):return leaf()
        return Expr(op,(a,b),None)
    for _ in range(100):
        e=rec(int(rng.integers(1,max_depth+1)))
        if e.depth()<=max_depth and e.nodes()<=max_nodes and e.op!='var':return e
    return Expr('diff',(leaf(),),None)

def structure_is_clean(e):
    if e.op in FORBIDDEN_TEMPLATE_OPS:return False
    return all(structure_is_clean(a) for a in e.args)

def program_signature(e):
    return e.text()

# ---------------- worlds ----------------
def _ar(r,n,rho,sd=1.0):
    x=np.zeros(n);innov=sd*math.sqrt(max(1-rho*rho,1e-6))
    x[0]=r.normal(scale=sd)
    for i in range(1,n):x[i]=rho*x[i-1]+r.normal(scale=innov)
    return x

def hidden_feature(family,env,meta):
    if family=='circular':
        d=env['x0']-base.lag(env['x0']);th=float(meta['wrap_threshold'])
        return d-(d>th).astype(float)+((-d)>th).astype(float)
    if family=='rank':
        left=base.expmean(base.lag(env['x1']),float(meta['filter_tau']))
        left=left-base.lag(left);gate=(base.weighted_rank(env['x0'],float(meta['rank_tau']))>float(meta['quantile'])).astype(float)
        return left*gate
    return np.zeros(len(next(iter(env.values()))))

def reference_expr(family,meta):
    if family=='circular':
        x=var('x0');dd=Expr('diff',(x,),None);pos=Expr('threshold',(dd,),float(meta['wrap_threshold']));neg=Expr('threshold',(Expr('negate',(dd,),None),),float(meta['wrap_threshold']))
        return Expr('add',(Expr('subtract',(dd,pos),None),neg),None)
    if family=='rank':
        x0=var('x0');x1=var('x1');left=Expr('diff',(Expr('filter_mean',(Expr('lag',(x1,),None),),float(meta['filter_tau'])),),None);gate=Expr('threshold',(Expr('rank',(x0,),float(meta['rank_tau'])),),float(meta['quantile']));return Expr('interaction',(left,gate),None)
    raise ValueError(family)

def make_world(family,seed,n=700,action_scale=1.0,meta=None):
    seed=int(seed);r=np.random.default_rng(seed);u=r.normal(size=(n,2))*float(action_scale);t=np.arange(n)
    if family in ('corrupt_circular','corrupt_rank'):
        bf='circular' if family=='corrupt_circular' else 'rank';env,y,h,m=make_world(bf,seed,n,action_scale,meta);rr=np.random.default_rng(seed+818181);shift=int(rr.integers(n//4,3*n//4));return env,np.roll(y,shift),np.zeros_like(h),{**m,'corruption':'circular_shift_y','shift':shift}
    if family=='circular':
        m=dict(meta or {});th=float(m.get('wrap_threshold',r.uniform(.43,.57)));th=th+.013 if abs(th-.5)<.01 else th;m={'wrap_threshold':th}
        drive=np.zeros(n)
        for i in range(1,n):drive[i]=.85*drive[i-1]+.22*r.normal()+.10*np.tanh(u[i,0])
        phase=np.cumsum(.22*np.tanh(drive));x0=np.mod(phase,1.0);env={'x0':x0,'x1':r.normal(size=n),'x2':.6*np.sin(.07*t)+.25*r.normal(size=n),'u0':u[:,0],'u1':u[:,1]}
        for j in range(3):env[f'd{j}']=_ar(np.random.default_rng(seed+700+j),n,.78,.5)
        h=hidden_feature('circular',env,m);y=1.8*h+.08*u[:,0]+.04*r.normal(size=n);return env,y,h,m
    if family=='rank':
        m=dict(meta or {});wf=float(m.get('filter_tau',r.uniform(2.2,5.8)));wr=float(m.get('rank_tau',r.uniform(6.2,14.8)));q=float(m.get('quantile',r.uniform(.58,.88)))
        if min(abs(wf-z) for z in (2.,3.,5.,9.))<.03:wf+=.071
        if min(abs(wr-z) for z in (2.,3.,5.,9.))<.03:wr+=.071
        if min(abs(q-z) for z in (.25,.5,.75))<.015:q+=.031
        m={'filter_tau':wf,'rank_tau':wr,'quantile':q};raw=np.zeros(n)
        for i in range(1,n):raw[i]=.88*raw[i-1]+.35*r.normal()
        x0=np.tanh(raw);x1=np.cumsum(.08*r.normal(size=n)+.025*u[:,1]);env={'x0':x0,'x1':x1,'x2':r.normal(size=n),'u0':u[:,0],'u1':u[:,1]};h=hidden_feature('rank',env,m);y=4.0*h+.18*u[:,1]+.04*r.normal(size=n);return env,y,h,m
    if family=='null':
        x0=np.zeros(n);x1=np.zeros(n);x2=np.zeros(n)
        for i in range(1,n):x0[i]=.7*x0[i-1]+.35*r.normal()+.1*u[i-1,0];x1[i]=.75*x1[i-1]+.32*r.normal()+.15*u[i-1,1];x2[i]=.6*x2[i-1]+.4*r.normal()
        env={'x0':x0,'x1':x1,'x2':x2,'u0':u[:,0],'u1':u[:,1]};h=np.zeros(n);y=.6*x1+.25*base.lag(x1)-.22*x0*x2+.20*base.movmean_discrete(x2,3)+.08*u[:,0]+.08*r.normal(size=n);return env,y,h,{}
    if family=='noise':
        env={'x0':r.normal(size=n),'x1':r.normal(size=n),'x2':r.normal(size=n),'u0':u[:,0],'u1':u[:,1]};return env,r.normal(size=n),np.zeros(n),{}
    if family=='spurious':
        rho=float((meta or {}).get('rho',.985));n=max(n,1200);r=np.random.default_rng(seed);u=r.normal(size=(n,2))*float(action_scale);env={'u0':u[:,0],'u1':u[:,1]}
        for k in ('x0','x1','x2','d0','d1','d2'):env[k]=_ar(np.random.default_rng(seed+100+hash(k)%97),n,rho,1.0)
        y=_ar(np.random.default_rng(seed+999),n,rho,1.0);return env,y,np.zeros(n),{'rho':rho}
    raise ValueError(family)

def b0_and_oracle(env,y,h):
    return base.b0_and_oracle(env,np.asarray(y,float),np.asarray(h,float))

def novelty_r2(z,base0):return base.novelty_r2(np.asarray(z,float),base0['X'],base0['tr'])

def extension_fit(base0,y,z):return base.extension_fit(base0,np.asarray(y,float),np.asarray(z,float))

# ---------------- search ----------------
def _var_weights(env,res,tr):
    keys=tuple(sorted(env));s=[]
    for k in keys:
        vals=[abs(base.corr(env[k][tr],res[tr])),abs(base.corr((env[k]-base.lag(env[k]))[tr],res[tr]))]
        s.append(max(vals)+.01)
    w=np.asarray(s,float);return keys,w/w.sum()

def candidate_bank(env,base0,seed,count=SEARCH_BUDGET,guided=False,shuffle=False):
    rng=np.random.default_rng(int(seed));res=np.array(base0['residual'],copy=True)
    if shuffle:res=base.block_permute(res,rng,20)
    keys=tuple(sorted(env));vw=None
    if guided:keys,vw=_var_weights(env,res,base0['tr'])
    out=[];seen=set()
    attempts=0
    while len(out)<int(count) and attempts<int(count)*30:
        attempts+=1;e=random_expr(keys,rng,var_weights=vw);e=tune_constants(e,env,res,base0['tr']);sig=e.text()
        if sig in seen:continue
        seen.add(sig);z=evaluate(e,env)
        if np.all(np.isfinite(z)) and np.max(np.abs(np.nan_to_num(z)))<1e9:out.append((e,z))
    return out,res

def select_guided(env,y,base0,seed,count=SEARCH_BUDGET,shuffle=False):
    bank,res=candidate_bank(env,base0,seed,count,guided=True,shuffle=shuffle);sc=[]
    for e,z in bank:
        fit=extension_fit(base0,y,z);rc=abs(base.corr(z[base0['tr']],res[base0['tr']]));sc.append((rc-MDL_LAMBDA*e.mdl_cost(),e,z,fit,rc))
    return max(sc,key=lambda q:q[0]),len(bank)

def select_controls(env,y,base0,seed,count=SEARCH_BUDGET):
    bank,res=candidate_bank(env,base0,seed,count,guided=False,shuffle=False);rows=[]
    for e,z in bank:
        fit=extension_fit(base0,y,z);rc=abs(base.corr(z[base0['tr']],res[base0['tr']]));rows.append((e,z,fit,rc))
    a=max(rows,key=lambda q:q[3]);b=min(rows,key=lambda q:q[2]['val_rmse']);return a,b,[x[0].text() for x in rows],len(bank)

def prechallenge(e,z,fit,base0):
    r2=novelty_r2(z,base0);gain=(base0['b0_val_rmse']-fit['val_rmse'])/max(base0['b0_val_rmse'],1e-12);return bool(r2<=NOVELTY_R2_MAX and gain>=VALIDATION_GAIN_MIN),float(r2),float(gain)

def acf_length(x,max_lag=250,threshold=.10,run=4):
    x=np.asarray(x,float);x=x-np.mean(x);den=float(np.dot(x,x));
    if den<1e-12:return 1
    good=0
    for k in range(1,min(max_lag,len(x)//3)+1):
        ac=abs(float(np.dot(x[:-k],x[k:])/den))
        good=good+1 if ac<threshold else 0
        if good>=run:return max(1,k-run+1)
    return min(max_lag,max(1,len(x)//4))

def challenge(e,family,meta,seed,B=999):
    env,y,h,m=make_world(family,int(seed),action_scale=1.6,meta=meta);b=b0_and_oracle(env,y,h);z=evaluate(e,env);fit=extension_fit(b,y,z);actual=b['b0_val_rmse']-fit['val_rmse'];tr=b['tr'];acl=max(acf_length(z[tr]),acf_length(b['residual'][tr]));block=min(max(20,int(math.ceil(1.25*acl))),max(20,len(tr)//4));rng=np.random.default_rng(int(seed)+404);null=[]
    for _ in range(int(B)):
        zp=base.block_permute(z,rng,block);null.append(b['b0_val_rmse']-extension_fit(b,y,zp)['val_rmse'])
    pv=(1+sum(x>=actual for x in null))/(B+1);return float(pv),float(actual),{'acf_length':int(acl),'block_length':int(block),'meta':m}

def admit(e,z,fit,base0,y,family,meta,seed,B=999):
    ok,r2,gain=prechallenge(e,z,fit,base0);d={'novelty_r2':r2,'validation_gain':gain,'challenge_p':1.0}
    if not ok:return False,d
    pv,cg,cd=challenge(e,family,meta,seed,B);d.update({'challenge_p':pv,'challenge_gain':cg,**cd});return bool(pv<=CHALLENGE_P_MAX),d

def run_seed_core(family,seed,challenge_B=999,search_budget=SEARCH_BUDGET):
    env,y,h,meta=make_world(family,int(seed));b=b0_and_oracle(env,y,h);actions=np.column_stack([env['u0'],env['u1']]);obs,thr,opened=base.calibrated_trigger(b['residual'][b['tr']],actions[b['tr']],int(seed)+101,B=min(199,challenge_B))
    # R-B: trigger is diagnostic only; every arm searches.
    gs,gn=select_guided(env,y,b,int(seed)+1001,search_budget,False);_,ge,gz,gfit,_=gs
    b2a,b2b,bank,bn=select_controls(env,y,b,int(seed)+2001,search_budget);b3s,b3n=select_guided(env,y,b,int(seed)+3001,search_budget,True);_,b3e,b3z,b3fit,_=b3s
    def arm(e,z,fit,off):
        ok,d=admit(e,z,fit,b,y,family,meta,int(seed)+off,B=challenge_B);return {'accepted':bool(ok),'program':e.text(),'signature':e.text(),'test_rmse':float(fit['test_rmse'] if ok else b['b0_test_rmse']),'novelty_r2':d['novelty_r2'],'validation_gain':d['validation_gain'],'challenge_p':d['challenge_p'],'mdl_cost':e.mdl_cost(),'constants':e.constants(),'diag':d,'expr':e,'z':z,'fit':fit}
    ga=arm(ge,gz,gfit,700001);aa=arm(*b2a[:3],710001);bb=arm(*b2b[:3],720001);ba=arm(b3e,b3z,b3fit,730001)
    return {'family':family,'seed':int(seed),'meta':meta,'env':env,'y':y,'h':h,'base':b,'trigger_open':bool(opened),'trigger_observed':float(obs),'trigger_threshold':float(thr),'trigger_is_gate':False,'guided':ga,'b2a':aa,'b2b':bb,'b3':ba,'random_bank_signature':bank,'search_counts':{'guided':gn,'b2a':bn,'b2b':bn,'b3':b3n}}

def l0_hidden_metrics(family,seed):
    env,y,h,meta=make_world(family,seed);b=b0_and_oracle(env,y,h);tr=b['tr'];A=np.column_stack([np.ones(len(tr)),b['X'][tr]]);coef=np.linalg.lstsq(A,h[tr],rcond=None)[0];pred=np.column_stack([np.ones(len(h)),b['X']])@coef;ss=np.sum((h[tr]-h[tr].mean())**2);r2=1-float(np.sum((h[tr]-pred[tr])**2))/max(float(ss),1e-12);nrmse=base.rmse(h[tr],pred[tr])/max(float(np.std(h[tr])),1e-12);return {'r2':float(r2),'nrmse':float(nrmse),'reference_depth':reference_expr(family,meta).depth(),'reference_nodes':reference_expr(family,meta).nodes(),'meta':meta}
