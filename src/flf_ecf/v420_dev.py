from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np

OLD_GRID_PERIODS=(1.0,2.0,2*math.pi);OLD_GRID_WINDOWS=(2.0,3.0,5.0,9.0);OLD_GRID_QUANTILES=(.25,.5,.75)
RIDGES=(1e-8,1e-6,1e-4,1e-2,.1);VALIDATION_GAIN_MIN=.05;NOVELTY_R2_MAX=.50;CHALLENGE_P_MAX=.01;MDL_CONSTANT_COST=.5;B2_CANDIDATES=25

@dataclass
class Program:
    kind:str;a:str;b:str|None=None;p:float|None=None;q:float|None=None
    def text(self):
        return {'moddiff':f'modulo(diff({self.a}),{self.p:.6g})','rankprod':f'product(diff({self.b}),threshold(rank({self.a},{self.p:.6g}),{self.q:.6g}))','mean':f'filter_mean({self.a},{self.p:.6g})','prod':f'product({self.a},{self.b})','diff':f'diff({self.a})','modraw':f'modulo({self.a},{self.p:.6g})'}[self.kind]
    def constants(self):return int(self.p is not None)+int(self.q is not None)
    def mdl_cost(self):return 1.0+MDL_CONSTANT_COST*self.constants()

def replace(p,**kw):d=asdict(p);d.update(kw);return Program(**d)
def lag(x,k=1):x=np.asarray(x,float);z=np.empty_like(x);z[:k]=x[0];z[k:]=x[:-k];return z
def movmean_discrete(x,w):
    x=np.asarray(x,float);o=np.empty_like(x);cs=np.cumsum(np.r_[0.,x])
    for i in range(len(x)):a=max(0,i-w+1);o[i]=(cs[i+1]-cs[a])/(i+1-a)
    return o
def expmean(x,w):
    x=np.asarray(x,float);a=1-math.exp(-1/max(float(w),1e-6));o=np.empty_like(x);o[0]=x[0]
    for i in range(1,len(x)):o[i]=(1-a)*o[i-1]+a*x[i]
    return o
def weighted_rank(x,w):
    x=np.asarray(x,float);n=len(x);L=max(4,int(math.ceil(3*float(w))));num=np.zeros(n);den=np.zeros(n)
    for k in range(L):
        wt=math.exp(-k/max(float(w),1e-6))
        if k==0:num+=wt;den+=wt
        else:num[k:]+=wt*(x[:-k]<=x[k:]);den[k:]+=wt
    return num/np.maximum(den,1e-12)
def block_permute(x,r,block=20):
    x=np.asarray(x);bs=[x[i:i+block] for i in range(0,len(x),block)];order=r.permutation(len(bs));return np.concatenate([bs[i] for i in order])[:len(x)]
def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3:return 0.0
    a=a[m];b=b[m]
    return 0.0 if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def eval_program(p,env):
    x=env[p.a]
    if p.kind=='moddiff':return np.mod(x-lag(x),p.p)
    if p.kind=='rankprod':return (env[p.b]-lag(env[p.b]))*(weighted_rank(x,p.p)>p.q)
    if p.kind=='mean':return expmean(x,p.p)
    if p.kind=='prod':return x*env[p.b]
    if p.kind=='diff':return x-lag(x)
    if p.kind=='modraw':return np.mod(x,p.p)
    raise ValueError(p.kind)

def make_world(family,seed,n=520,action_scale=1.0,difficulty=1.0):
    r=np.random.default_rng(seed);u=r.normal(size=(n,2))*action_scale;x1=np.zeros(n);x2=np.zeros(n)
    for t in range(1,n):x1[t]=.75*x1[t-1]+.32*r.normal()+.15*u[t-1,1];x2[t]=.6*x2[t-1]+.4*r.normal()
    meta={}
    if family in ('circular','corrupt_circular'):
        P=float(r.uniform(.72,1.38));P=P+.071 if min(abs(P-z) for z in OLD_GRID_PERIODS)<.04 else P;phase=np.zeros(n);drive=.16*u[:,0]+.07*np.tanh(x1)+.11*r.normal(size=n)
        for t in range(1,n):phase[t]=phase[t-1]+drive[t]
        x0=np.mod(phase,P);h=np.mod(x0-lag(x0),P);env={'x0':x0,'x1':x1,'x2':x2,'u0':u[:,0],'u1':u[:,1]};rr=np.random.default_rng(seed+777)
        for j in range(4):
            z=np.zeros(n)
            for t in range(1,n):z[t]=.75*z[t-1]+.35*rr.normal()
            env[f'd{j}']=z
        y=.35*x1+.18*lag(x1)+.12*x2*lag(x1)+difficulty*1.4*h+.05*u[:,1]+.08*r.normal(size=n);meta={'period':P}
    elif family in ('rank','corrupt_rank'):
        x0=np.zeros(n)
        for t in range(1,n):x0[t]=.8*x0[t-1]+.3*r.normal()+.08*u[t-1,0]
        w=float(r.uniform(6.4,14.6));w=w+.71 if abs(w-9)<.3 else w;q=float(r.uniform(.58,.88));q=q+.041 if abs(q-.75)<.025 else q;h=(x1-lag(x1))*(weighted_rank(x0,w)>q);env={'x0':x0,'x1':x1,'u0':u[:,0],'u1':u[:,1]};y=.35*x1+.15*lag(x1)+difficulty*2.0*h+.06*u[:,0]+.08*r.normal(size=n);meta={'window':w,'quantile':q}
    elif family=='null':
        x0=np.zeros(n)
        for t in range(1,n):x0[t]=.7*x0[t-1]+.35*r.normal()+.1*u[t-1,0]
        env={'x0':x0,'x1':x1,'x2':x2,'u0':u[:,0],'u1':u[:,1]};h=np.zeros(n);y=.6*x1+.25*lag(x1)-.22*x0*x2+.20*movmean_discrete(x2,3)+.08*u[:,0]+.08*r.normal(size=n)
    elif family=='noise':env={'x0':r.normal(size=n),'x1':r.normal(size=n),'x2':r.normal(size=n),'u0':u[:,0],'u1':u[:,1]};h=np.zeros(n);y=r.normal(size=n)
    else:raise ValueError(family)
    if family.startswith('corrupt_'):y=block_permute(y,r,20)
    return env,np.asarray(y),np.asarray(h),meta

def splits(n):return np.arange(0,int(.55*n)),np.arange(int(.55*n),int(.75*n)),np.arange(int(.75*n),n)
def l0_matrix(env):
    cols=[];names=[];keys=sorted(env)
    for k in keys:
        x=env[k]
        for name,z in ((k,x),(f'lag1({k})',lag(x)),(f'diff1({k})',x-lag(x)),(f'mean2({k})',movmean_discrete(x,2)),(f'mean3({k})',movmean_discrete(x,3)),(f'mean5({k})',movmean_discrete(x,5))):cols.append(z);names.append(name)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:cols.append(env[a]*env[b]);names.append(f'product({a},{b})')
    return np.column_stack(cols),names
def fit_linear(X,y,idx,ridge):
    mu=X[idx].mean(0);sd=X[idx].std(0)+1e-8;Z=(X-mu)/sd;A=np.column_stack([np.ones(len(idx)),Z[idx]]);beta=np.linalg.solve(A.T@A+ridge*np.eye(A.shape[1]),A.T@y[idx]);return beta,mu,sd
def predict(m,X):b,mu,sd=m;Z=(X-mu)/sd;return np.column_stack([np.ones(len(X)),Z])@b
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def fit_complete(X,y,tr,val):
    ms=[fit_linear(X,y,tr,r) for r in RIDGES];return min(ms,key=lambda m:rmse(y[val],predict(m,X)[val]))
def extension_fit(base,y,z):
    tr,val,te=base['tr'],base['val'],base['te'];mu=float(np.mean(z[tr]));sd=float(np.std(z[tr])+1e-8);zz=(z-mu)/sd;res=base['residual'];beta=float(np.dot(zz[tr],res[tr])/max(np.dot(zz[tr],zz[tr]),1e-12));pp=base['pred']+beta*zz;return {'mu':mu,'sd':sd,'beta':beta,'pred':pp,'val_rmse':rmse(y[val],pp[val]),'test_rmse':rmse(y[te],pp[te]),'z':zz}
def b0_and_oracle(env,y,h):
    X,names=l0_matrix(env);tr,val,te=splits(len(y));b0=fit_complete(X,y,tr,val);bp=predict(b0,X);b={'X':X,'names':names,'tr':tr,'val':val,'te':te,'b0':b0,'pred':bp,'residual':y-bp,'b0_val_rmse':rmse(y[val],bp[val]),'b0_test_rmse':rmse(y[te],bp[te])};o=extension_fit(b,y,h);b['oracle']=o;b['b1_test_rmse']=o['test_rmse'];return b
def novelty_r2(z,X,tr):
    idx=np.asarray(tr);A=np.column_stack([np.ones(len(idx)),X[idx]]);beta=np.linalg.lstsq(A,z[idx],rcond=None)[0];p=A@beta;yy=z[idx];return 1-float(np.sum((yy-p)**2))/max(float(np.sum((yy-yy.mean())**2)),1e-12)

def sample_program(env,r):
    keys=sorted(env);kind=int(r.integers(6));a=keys[int(r.integers(len(keys)))]
    if kind==0:return Program('moddiff',a,p=float(r.uniform(.5,1.6)))
    if kind==1:return Program('rankprod',a,b=keys[int(r.integers(len(keys)))],p=float(r.uniform(4,16)),q=float(r.uniform(.4,.9)))
    if kind==2:return Program('mean',a,p=float(r.uniform(1.5,10)))
    if kind==3:return Program('prod',a,b=keys[int(r.integers(len(keys)))])
    if kind==4:return Program('diff',a)
    return Program('modraw',a,p=float(r.uniform(.5,1.6)))
def tune_constants(p,env,res,tr):
    def score(q):return abs(corr(eval_program(q,env)[tr],res[tr]))
    b=p
    if p.p is not None:
        grid=np.linspace(.5,1.6,13) if p.kind in ('moddiff','modraw') else (np.linspace(4.2,15.8,13) if p.kind=='rankprod' else np.linspace(1.6,9.8,11));b=max((replace(b,p=float(v)) for v in grid),key=score)
    if p.q is not None:b=max((replace(b,q=float(v)) for v in np.linspace(.42,.88,13)),key=score)
    return b
def candidate_bank(env,base,seed,count=B2_CANDIDATES,guided=False,shuffle=False):
    r=np.random.default_rng(seed);res=np.array(base['residual'],copy=True);res=block_permute(res,r,17) if shuffle else res;weights=None
    if guided:
        ks=sorted(env);w=np.asarray([abs(corr((env[k]-lag(env[k]))[base['tr']],res[base['tr']]))+.02 for k in ks]);weights=w/w.sum()
    out=[]
    for _ in range(count):
        p=sample_program(env,r)
        if guided and r.random()<.65:p=replace(p,a=str(r.choice(sorted(env),p=weights)))
        p=tune_constants(p,env,res,base['tr']);out.append((p,eval_program(p,env)))
    return out,res
def select_candidate(env,y,base,seed,mode,count=B2_CANDIDATES):
    bank,res=candidate_bank(env,base,seed,count,guided=mode in ('guided','shuffled'),shuffle=mode=='shuffled');sc=[]
    for p,z in bank:
        m=extension_fit(base,y,z);rs=abs(corr(z[base['tr']],res[base['tr']]));key=-rs if mode in ('guided','b2a','shuffled') else m['val_rmse'];sc.append((key,p,z,m,rs))
    return min(sc,key=lambda x:x[0])

def residual_score(resid,actions):
    r=np.asarray(resid,float);ac=abs(corr(r[1:],r[:-1]));inc=max(abs(corr(r,actions[:,j])) for j in range(actions.shape[1]));Y=r[2:];A=np.column_stack([np.ones(len(Y)),r[1:-1],r[:-2]]);b=np.linalg.lstsq(A,Y,rcond=None)[0];comp=max(0.,1-np.var(Y-A@b)/max(np.var(Y),1e-12));return ac+inc+comp
def calibrated_trigger(resid,actions,seed,B=199):
    r=np.random.default_rng(seed);null=[residual_score(block_permute(resid,r,20),actions) for _ in range(B)];thr=float(np.quantile(null,.99));obs=residual_score(resid,actions);return obs,thr,obs>=thr
def challenge_p(program,seed,family,difficulty=1.0,B=199):
    env,y,h,_=make_world(family,seed,action_scale=1.6,difficulty=difficulty);base=b0_and_oracle(env,y,h);z=eval_program(program,env);m=extension_fit(base,y,z);actual=base['b0_val_rmse']-m['val_rmse'];r=np.random.default_rng(seed+404);null=[]
    for _ in range(B):null.append(base['b0_val_rmse']-extension_fit(base,y,block_permute(z,r,20))['val_rmse'])
    return (1+sum(v>=actual for v in null))/(B+1),actual
def admission(program,z,m,base,y,challenge_seed,family):
    r2=novelty_r2(z,base['X'],base['tr']);gain=(base['b0_val_rmse']-m['val_rmse'])/max(base['b0_val_rmse'],1e-12)
    if r2>NOVELTY_R2_MAX or gain<VALIDATION_GAIN_MIN:return False,{'novelty_r2':r2,'validation_gain':gain,'challenge_p':1.0}
    p,cg=challenge_p(program,challenge_seed,family);return p<=CHALLENGE_P_MAX,{'novelty_r2':r2,'validation_gain':gain,'challenge_p':p,'challenge_gain':cg}
def bootstrap_ci(x,B=4000,seed=991):
    x=np.asarray(x,float);r=np.random.default_rng(seed);v=[float(np.mean(x[r.integers(0,len(x),len(x))])) for _ in range(B)];return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def instrument_seed(iteration,index):
    b=120000+iteration*10000;i=int(index);row={'iteration':iteration,'index':i}
    env,y,h,_=make_world('null',b+i);X,_=l0_matrix(env);tr,val,te=splits(len(y));m=fit_complete(X,y,tr,val);c=fit_linear(X,y,tr,1e-8);ce=rmse(y[te],predict(c,X)[te]);row['R1_relative_gap']=(rmse(y[te],predict(m,X)[te])-ce)/max(ce,1e-12)
    for fam in ('corrupt_circular','corrupt_rank'):
        env,y,h,_=make_world(fam,b+1000+i);z=b0_and_oracle(env,y,h);row[f'R2_{fam}_oracle_gain']=(z['b0_test_rmse']-z['b1_test_rmse'])/max(z['b0_test_rmse'],1e-12)
    bad=0
    for fam in ('circular','rank'):
        _,_,_,meta=make_world(fam,b+2000+i)
        if fam=='circular':bad+=any(abs(meta['period']-x)<1e-9 for x in OLD_GRID_PERIODS)
        else:bad+=any(abs(meta['window']-x)<1e-9 for x in OLD_GRID_WINDOWS)+any(abs(meta['quantile']-x)<1e-9 for x in OLD_GRID_QUANTILES)
    row['R3_grid_matches']=int(bad)
    for fam in ('circular','rank'):
        env,y,h,_=make_world(fam,b+3000+i);z=b0_and_oracle(env,y,h);og=(z['b0_test_rmse']-z['b1_test_rmse'])/max(z['b0_test_rmse'],1e-12);q=select_candidate(env,y,z,b+53000+i,'b2b');gap=(z['b0_test_rmse']-q[3]['test_rmse'])/max(z['b0_test_rmse']-z['b1_test_rmse'],1e-12);row[f'R4_{fam}_oracle_gain']=og;row[f'R4_{fam}_B2_success']=bool(gap>=.35);row[f'R4_{fam}_B2_gap']=gap
    env,y,h,_=make_world('null',b+7000+i);z=b0_and_oracle(env,y,h);actions=np.column_stack([env['u0'],env['u1']]);obs,thr,op=calibrated_trigger(z['residual'],actions,b+8000+i);row['R6_null_trigger_open']=bool(op);row['R6_null_accepted']=False
    if op:
        q=select_candidate(env,y,z,b+9000+i,'guided');ok,ad=admission(q[1],q[2],q[3],z,y,b+900000+i,'null');row['R6_null_accepted']=bool(ok);row['R6_diagnostics']=ad
    return row
def noise_trigger_row(iteration,index):
    b=160000+iteration*10000;i=int(index);env,y,h,_=make_world('noise',b+i);z=b0_and_oracle(env,y,h);actions=np.column_stack([env['u0'],env['u1']]);obs,thr,op=calibrated_trigger(z['residual'],actions,b+5000+i);return {'iteration':iteration,'index':i,'open':bool(op),'score':obs,'threshold':thr}
