import numpy as np, math, random, statistics
from dataclasses import dataclass,asdict
from functools import lru_cache

@dataclass(frozen=True)
class G:
 q:int;w:int;f:int;ks:int;ka:int;norm:int;target:int;policy:int;actmode:int;mdl:int
Q=[None,.5,.65,.78,.88];MW=[0,.006,.015,.03]
def rg(r):return G(r.randrange(5),r.randrange(1,5),r.randrange(4),r.randrange(3,9),r.randrange(3,9),r.randrange(2),r.randrange(2),r.randrange(2),r.randrange(2),r.randrange(4))
def mut(g,r):
 a=list(asdict(g).values());j=r.randrange(10);bounds=[(0,4),(1,4),(0,3),(3,8),(3,8),(0,1),(0,1),(0,1),(0,1),(0,3)];lo,hi=bounds[j]
 if lo==0 and hi==1:a[j]=1-a[j]
 else:a[j]=max(lo,min(hi,a[j]+r.choice([-1,1])))
 return G(*a)
def cross(a,b,r):
 av=list(asdict(a).values());bv=list(asdict(b).values());cut=r.randrange(1,10);return G(*(av[:cut]+bv[cut:]))

class World:
 def __init__(self,seed,noise=False):
  self.r=np.random.default_rng(seed);self.noise=noise;self.z=int(self.r.integers(6));q,_=np.linalg.qr(self.r.normal(size=(3,3)));base=np.array([[3*math.cos(i*math.pi/3),3*math.sin(i*math.pi/3),(i-2.5)*.6] for i in range(6)]);self.c=base@q;self.pref=self.r.permutation(6);self.delta=np.array([-2,-1,0,1,2,3])
 def sec(self,u):a=(math.atan2(float(u[1]),float(u[0]))+2*math.pi)%(2*math.pi);return int(a/(2*math.pi)*6)%6
 def step(self,u):
  if self.noise:return self.r.normal(size=3),u,self.r.normal(size=3),float(self.r.random()<.5)
  z0=self.z;x=self.c[z0]+.045*self.r.normal(size=3);s=self.sec(u);rw=1.0 if s==int(self.pref[z0]) else (.1 if s==(int(self.pref[z0])+1)%6 else 0.0)
  self.z=(z0+int(self.delta[s]))%6 if self.r.random()<.995 else int(self.r.integers(6));y=self.c[self.z]+.045*self.r.normal(size=3);return x,u,y,rw
 def roll(self,n,s):
  rr=np.random.default_rng(s);o=[]
  for _ in range(n):
   a=float(rr.uniform(0,2*math.pi));u=np.array([math.cos(a),math.sin(a)])*float(rr.uniform(.75,1.25));o.append(self.step(u))
  return o

def km(X,k):
 X=np.asarray(X);C=[X[0]]
 while len(C)<k:
  D=np.min([((X-c)**2).sum(1) for c in C],axis=0);C.append(X[int(np.argmax(D))])
 C=np.array(C)
 for _ in range(8):
  lab=((X[:,None]-C[None])**2).sum(2).argmin(1);N=C.copy()
  for j in range(k):
   if np.any(lab==j):N[j]=X[lab==j].mean(0)
  if np.allclose(N,C):break
  C=N
 return C

def ft(xs,i,w,m):
 H=xs[max(0,i-w+1):i+1];last=H[-1];mean=H.mean(0);d=H[-1]-H[0];return(last,mean,d,np.r_[last,mean,d])[m]

class Model:
 def __init__(self,g):self.g=g
 def fit(self,rows):
  xs=np.array([z[0] for z in rows]);us=np.array([z[1] for z in rows]);ys=np.array([z[2] for z in rows]);rs=np.array([z[3] for z in rows]);jump=np.linalg.norm(ys-xs,axis=1);idx=np.arange(len(rows)) if Q[self.g.q] is None else np.where(jump>=np.quantile(jump,Q[self.g.q]))[0]
  if len(idx)<30:idx=np.arange(len(rows))
  F=np.array([ft(xs,i,self.g.w,self.g.f) for i in idx]);A=us[idx];T=(ys[idx]-xs[idx]) if self.g.target else ys[idx];R=rs[idx]
  if self.g.norm:self.mu=F.mean(0);self.sd=F.std(0)+1e-6;F=(F-self.mu)/self.sd;self.tm=T.mean(0);self.ts=T.std(0)+1e-6;Tn=(T-self.tm)/self.ts
  else:self.mu=np.zeros(F.shape[1]);self.sd=np.ones(F.shape[1]);self.tm=np.zeros(3);self.ts=np.ones(3);Tn=T
  self.cs=km(F,self.g.ks);sl=((F[:,None]-self.cs[None])**2).sum(2).argmin(1);self.n=len(idx);self.pt=np.zeros((self.g.ks,self.g.ka,3));self.pr=np.zeros((self.g.ks,self.g.ka));self.ac=np.zeros((self.g.ks,self.g.ka,2));gt=Tn.mean(0);gr=R.mean()
  if self.g.actmode==0:global_ca=km(A,self.g.ka)
  for s in range(self.g.ks):
   sm=np.where(sl==s)[0]
   if len(sm)==0:self.ac[s]=global_ca if self.g.actmode==0 else np.repeat(A.mean(0)[None,:],self.g.ka,axis=0);continue
   if self.g.actmode==0:ca=global_ca;al=((A[sm,None]-ca[None])**2).sum(2).argmin(1)
   else:
    tc=km(Tn[sm],self.g.ka);al=((Tn[sm,None]-tc[None])**2).sum(2).argmin(1);ca=np.zeros((self.g.ka,2))
    for a in range(self.g.ka):ca[a]=A[sm[al==a]].mean(0) if np.any(al==a) else A[sm].mean(0)
   self.ac[s]=ca
   for a in range(self.g.ka):
    ids=sm[al==a];self.pt[s,a]=Tn[ids].mean(0) if len(ids) else gt;self.pr[s,a]=R[ids].mean() if len(ids) else gr
  return self
 def state(self,h):
  z=ft(np.asarray(h),len(h)-1,self.g.w,self.g.f);z=(z-self.mu)/self.sd if self.g.norm else z;return int(((self.cs-z)**2).sum(1).argmin())
 def ar(self,h,u):s=self.state(h);return s,int(((self.ac[s]-u)**2).sum(1).argmin())
 def pred(self,h,u):
  s,a=self.ar(h,u);t=self.pt[s,a];t=t*self.ts+self.tm if self.g.norm else t;return np.asarray(h[-1])+t if self.g.target else t
 def reward(self,h,u):s,a=self.ar(h,u);return self.pr[s,a]
 def choose(self,h):
  s=self.state(h);a=int(self.pr[s].argmax());p=self.ac[s,a]
  if self.g.policy==1:p=p/(np.linalg.norm(p)+1e-9)
  return p

def met(m,rows):
 xs=[z[0] for z in rows];se=[];re=[]
 for i,(x,u,y,r) in enumerate(rows):
  h=xs[max(0,i-m.g.w+1):i+1];p=m.pred(h,u);se.append(float(((p-y)**2).mean()));re.append((m.reward(h,u)-r)**2)
 return math.sqrt(statistics.mean(se)),math.sqrt(statistics.mean(re))
def ctl(seed,m,n=120):
 w=World(seed);h=[];tot=0
 for _ in range(m.g.w):x,u,y,r=w.step(np.array([1.,0.]));h.append(y)
 for _ in range(n):u=m.choose(h);x,u,y,r=w.step(u);tot+=r;h.append(y);h=h[-m.g.w:]
 return tot/n

DATA={}
for s in range(10):
 a=World(1000+s).roll(360,3000+s);DATA[s]=(a[:245],a[245:])
NO={}
for s in range(2):
 a=World(9000+s,True).roll(300,10000+s);NO[s]=(a[:205],a[205:])
def shuf(rows,seed):
 rr=random.Random(seed);aa=[z[1].copy() for z in rows];rr.shuffle(aa);return[(z[0],aa[i],z[2],z[3]) for i,z in enumerate(rows)]

@lru_cache(None)
def obj(g):
 ps=[];rs=[];cs=[]
 for s in (0,1,2):
  m=Model(g).fit(DATA[s][0]);p,r=met(m,DATA[s][1]);ps.append(p);rs.append(r);cs.append((g.ks*g.ka+m.n/len(DATA[s][0]))/100+MW[g.mdl]*(g.ks+g.ka+g.w+g.actmode))
 c=statistics.mean(ctl(1000+s,Model(g).fit(DATA[s][0]),70) for s in (1,2))
 return statistics.mean(ps),statistics.mean(rs),statistics.mean(cs),c,statistics.pstdev(ps)
def finalscore(g):
 p,r,c,u,v=obj(g);return -p-.18*r-.07*c+.55*u-.10*v
def rank_select(pop,n=4):
 uniq=list(dict.fromkeys(pop));vals={g:obj(g) for g in uniq};ranks={g:0.0 for g in uniq};specs=[(0,False),(1,False),(2,False),(3,True),(4,False)]
 for j,rev in specs:
  order=sorted(uniq,key=lambda g:vals[g][j],reverse=rev)
  for i,g in enumerate(order):ranks[g]+=i
 return sorted(uniq,key=lambda g:(ranks[g],-finalscore(g)))[:n]
def evolve(seed,budget=48,mutation=True,crossover=True):
 r=random.Random(seed);pop=[];seen=set()
 while len(pop)<10:
  x=rg(r)
  if x not in seen:seen.add(x);pop.append(x)
 initial=max(finalscore(x) for x in pop);ev=len(seen)
 while ev<budget:
  elite=rank_select(pop,4);new=elite[:];tries=0
  while len(new)<10 and ev<budget and tries<120:
   c=cross(r.choice(elite),r.choice(elite),r) if crossover else r.choice(elite)
   if mutation:c=mut(c,r)
   tries+=1
   if c not in seen:seen.add(c);ev+=1;new.append(c)
  if len(new)==len(elite):break
  while len(new)<10:new.append(r.choice(elite))
  pop=new
 best=max(seen,key=finalscore);return best,finalscore(best),initial,ev
def held(g):
 ps=[];ct=[]
 for s in (3,4,5):ps.append(met(Model(g).fit(DATA[s][0]),DATA[s][1])[0])
 for s in (6,7):ct.append(ctl(1000+s,Model(g).fit(DATA[s][0])))
 return statistics.mean(ps),statistics.mean(ct)

def run_once(seed,evolution_budget=30,random_budget=30,run_ablations=True):
 fixed=G(3,2,0,4,4,1,0,0,0,1);fp,fc=held(fixed);g,es,init,ev=evolve(20000+seed,evolution_budget);hp,hc=held(g);rr=random.Random(30000+seed);pool=[];seen=set()
 while len(pool)<random_budget:
  x=rg(rr)
  if x not in seen:seen.add(x);pool.append(x)
 rb=max(pool,key=finalscore);rp,rc=held(rb);sh=[];no=[]
 for s in (3,4):sh.append(met(Model(g).fit(shuf(DATA[s][0],60000+seed+s)),DATA[s][1])[0])
 for j in range(2):no.append(met(Model(g).fit(NO[j][0]),NO[j][1])[0])
 row={'seed':seed,'genome':asdict(g),'evals':ev,'evolution_score':es,'initial_best_score':init,'heldout_rmse':hp,'heldout_control':hc,'fixed_rmse':fp,'fixed_control':fc,'random_rmse':rp,'random_control':rc,'shuffled_rmse':statistics.mean(sh),'noise_rmse':statistics.mean(no)}
 if run_ablations:
  _,nm,_,_=evolve(40000+seed,20,False,True);_,nc,_,_=evolve(50000+seed,20,True,False);row['no_mutation_score']=nm;row['no_crossover_score']=nc
 return row

def summarize(rows,ablation_rows):
 fixed=G(3,2,0,4,4,1,0,0,0,1);fp,fc=held(fixed);mean=lambda k:statistics.mean(x[k] for x in rows)
 s={'main_runs':len(rows),'ablation_runs':len(ablation_rows),'heldout_rmse':mean('heldout_rmse'),'fixed_rmse':fp,'random_search_rmse':mean('random_rmse'),'heldout_control':mean('heldout_control'),'fixed_control':fc,'random_search_control':mean('random_control'),'evolution_score':mean('evolution_score'),'initial_best_score':mean('initial_best_score'),'shuffled_rmse':mean('shuffled_rmse'),'noise_rmse':mean('noise_rmse')}
 s['no_mutation_score']=statistics.mean(x['no_mutation_score'] for x in ablation_rows);s['no_crossover_score']=statistics.mean(x['no_crossover_score'] for x in ablation_rows);s['prediction_gain_vs_fixed']=(fp-s['heldout_rmse'])/fp;s['prediction_gain_vs_random']=(s['random_search_rmse']-s['heldout_rmse'])/s['random_search_rmse'];s['control_gain_vs_fixed']=s['heldout_control']-fc;s['evolution_gain']=s['evolution_score']-s['initial_best_score'];ab_evo=statistics.mean(x['evolution_score'] for x in ablation_rows);s['mutation_gap']=ab_evo-s['no_mutation_score'];s['crossover_gap']=ab_evo-s['no_crossover_score'];s['shuffle_degradation']=(s['shuffled_rmse']-s['heldout_rmse'])/s['heldout_rmse'];s['noise_degradation']=(s['noise_rmse']-s['heldout_rmse'])/s['heldout_rmse'];medp=statistics.median(x['heldout_rmse'] for x in rows);medc=statistics.median(x['heldout_control'] for x in rows);s['independent_rediscovery']=statistics.mean((x['heldout_rmse']<=1.2*medp and abs(x['heldout_control']-medc)<=.10) for x in rows);s['gate_pass']=(s['prediction_gain_vs_fixed']>=.08 and s['prediction_gain_vs_random']>=.03 and s['control_gain_vs_fixed']>=.08 and s['evolution_gain']>=.03 and s['mutation_gap']>=.005 and s['crossover_gap']>=-.005 and s['shuffle_degradation']>=.50 and s['noise_degradation']>=1.00 and s['independent_rediscovery']>=.80);return s

def benchmark(main_runs=5,ablation_runs=3,evolution_budget=30,random_budget=30):
 rows=[run_once(seed,evolution_budget,random_budget,seed<ablation_runs) for seed in range(main_runs)];return summarize(rows,rows[:ablation_runs]),rows
