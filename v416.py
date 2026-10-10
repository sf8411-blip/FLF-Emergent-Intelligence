from __future__ import annotations
import math, random, statistics, itertools
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable

Event = Tuple[float, int]
Step = Tuple[int, List[int]]  # (anonymous action token, burst of anonymous event tokens)

@dataclass
class WorldSpec:
    states: int = 6
    aliases_per_state: int = 4
    actions: int = 3
    noise_tokens: int = 10
    alias_emit_prob: float = 0.80
    noise_emit_prob: float = 0.35

class AnonymousEventWorld:
    """Synthetic benchmark world. Hidden state semantics are NEVER passed to learner."""
    def __init__(self, seed: int, spec: WorldSpec = WorldSpec()):
        self.rng = random.Random(seed)
        self.spec = spec
        toks = list(range(spec.states * spec.aliases_per_state + spec.noise_tokens))
        self.rng.shuffle(toks)
        self.aliases = [toks[i*spec.aliases_per_state:(i+1)*spec.aliases_per_state] for i in range(spec.states)]
        self.noise = toks[spec.states*spec.aliases_per_state:]
        self.action_tokens = list(range(100, 100+spec.actions))
        self.rng.shuffle(self.action_tokens)
        shifts = [1, 2, -1][:spec.actions]
        self.transition = {}
        for a_idx, atok in enumerate(self.action_tokens):
            sh = shifts[a_idx]
            for s in range(spec.states):
                n = (s + sh) % spec.states
                self.transition[(s, atok)] = n
        self.state = self.rng.randrange(spec.states)

    def burst(self) -> List[int]:
        out=[]
        aliases=self.aliases[self.state]
        out.append(self.rng.choice(aliases))
        if self.rng.random() < self.spec.alias_emit_prob:
            second=self.rng.choice([x for x in aliases if x!=out[0]])
            out.append(second)
        if self.rng.random() < self.spec.noise_emit_prob:
            out.append(self.rng.choice(self.noise))
        self.rng.shuffle(out)
        return out

    def raw_rollout(self, steps: int, policy_seed: int) -> List[Event]:
        """Return one timestamped stream. Action tokens and sensory tokens share one token space.
        The learner only knows which opaque tokens it is physically able to emit as interventions.
        """
        pr = random.Random(policy_seed)
        stream=[]; t=0.0
        for _ in range(steps):
            action=pr.choice(self.action_tokens)
            stream.append((t, action))
            obs=self.burst()
            dt=0.01
            for tok in obs:
                dt += pr.uniform(0.002,0.018)
                stream.append((t+dt, tok))
            self.state=self.transition[(self.state, action)]
            t += pr.uniform(0.65,1.35)
        return stream


def reconstruct_steps(stream: List[Event], intervention_tokens: Iterable[int]) -> List[Step]:
    """Infer experiment episodes from a raw stream using only self-intervention provenance.
    No entity IDs, Boolean values, state labels, or pre-grouped observations are supplied.
    """
    acts=set(intervention_tokens); data=[]; current_action=None; obs=[]
    for _,tok in sorted(stream):
        if tok in acts:
            if current_action is not None and obs:data.append((current_action,obs))
            current_action=tok; obs=[]
        elif current_action is not None:
            obs.append(tok)
    if current_action is not None and obs:data.append((current_action,obs))
    return data


def js_divergence(p: Dict[int,float], q: Dict[int,float]) -> float:
    keys=set(p)|set(q)
    m={k:0.5*p.get(k,0.0)+0.5*q.get(k,0.0) for k in keys}
    def kl(a,b):
        s=0.0
        for k,v in a.items():
            if v>0 and b.get(k,0)>0: s += v*math.log(v/b[k],2)
        return s
    return 0.5*kl(p,m)+0.5*kl(q,m)


def normalize(c: Counter) -> Dict[int,float]:
    total=sum(c.values())
    if total==0:return {}
    return {k:v/total for k,v in c.items()}


def token_profiles(data: List[Step]) -> Dict[int, Dict[Tuple[int,int],float]]:
    counts=defaultdict(Counter)
    for i in range(len(data)-1):
        a,cur=data[i]; _,nxt=data[i+1]
        for t in cur:
            for nt in nxt: counts[t][(a,nt)] += 1
    return {t:normalize(c) for t,c in counts.items()}


def cooccur_profiles(data: List[Step]) -> Dict[int, Dict[int,float]]:
    counts=defaultdict(Counter)
    for _,burst in data:
        for t in burst:
            for u in burst:
                if u!=t: counts[t][u]+=1
    return {t:normalize(c) for t,c in counts.items()}


def distance_matrix(data: List[Step]) -> Tuple[List[int], Dict[Tuple[int,int],float]]:
    pp=token_profiles(data); cp=cooccur_profiles(data)
    tokens=sorted(set(t for _,b in data for t in b))
    D={}
    for i,t in enumerate(tokens):
        for u in tokens[i+1:]:
            pred=js_divergence(pp.get(t,{}), pp.get(u,{}))
            ct=cp.get(t,{}).get(u,0.0); cu=cp.get(u,{}).get(t,0.0)
            co=1.0-min(1.0, 2.0*(ct+cu))
            D[(t,u)] = 0.45*pred + 0.55*co
    return tokens,D


def mapping_from_distance(tokens, D, threshold: float) -> Dict[int,int]:
    parent={t:t for t in tokens}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for (t,u),d in D.items():
        if d<=threshold: union(t,u)
    roots={}; mapping={}
    for t in tokens:
        r=find(t)
        if r not in roots:roots[r]=len(roots)
        mapping[t]=roots[r]
    return mapping


def nll_for_mapping(data: List[Step], mapping: Dict[int,int], action_conditioned: bool=True) -> float:
    trans=defaultdict(Counter)
    n=0
    for i in range(len(data)-1):
        a,cur=data[i];_,nxt=data[i+1]
        c=Counter(mapping[t] for t in cur if t in mapping)
        d=Counter(mapping[t] for t in nxt if t in mapping)
        if not c or not d:continue
        cs=c.most_common(1)[0][0]; ns=d.most_common(1)[0][0]
        key=(cs,a) if action_conditioned else (cs,None)
        trans[key][ns]+=1;n+=1
    k=max(1,len(set(mapping.values())))
    loss=0.0
    for key,cnt in trans.items():
        tot=sum(cnt.values())+k
        for v in cnt.values(): loss -= v*math.log((v+1)/tot)
        unseen=k-len(cnt)
        if unseen>0: loss -= unseen*math.log(1/tot)*0.05
    return loss/max(1,n)


def reconstruction_cost(data: List[Step], mapping: Dict[int,int]) -> float:
    freq=Counter(t for _,b in data for t in b if t in mapping)
    members=defaultdict(list)
    for t,c in mapping.items():members[c].append(t)
    total=sum(freq.values())
    loss=0.0
    for c,ts in members.items():
        z=sum(freq[t] for t in ts)
        if z==0:continue
        for t in ts:
            if freq[t]:
                p=freq[t]/z
                loss -= (freq[t]/total)*math.log(max(p,1e-12))
    return loss


def mdl_mapping(data: List[Step]) -> Dict[int,int]:
    best=None
    tokens,D=distance_matrix(data)
    ntok=len(tokens)
    for th in [i/40 for i in range(2,31)]:
        m=mapping_from_distance(tokens,D,th)
        k=len(set(m.values()))
        pred=nll_for_mapping(data,m,True)
        recon=reconstruction_cost(data,m)
        cost=pred + 0.55*recon + 0.008*k*math.log(max(2,ntok))
        cand=(cost,m)
        if best is None or cand[0]<best[0]:best=cand
    return best[1]


def pair_equivalence(mapping: Dict[int,int]) -> set[Tuple[int,int]]:
    toks=sorted(mapping)
    return {(a,b) for i,a in enumerate(toks) for b in toks[i+1:] if mapping[a]==mapping[b]}


def independent_mapping(data: List[Step]) -> Dict[int,int]:
    mid=len(data)//2
    m1=mdl_mapping(data[:mid]);m2=mdl_mapping(data[mid:])
    common=set(m1)&set(m2)
    e1=pair_equivalence({t:m1[t] for t in common});e2=pair_equivalence({t:m2[t] for t in common})
    stable=e1 & e2
    parent={t:t for t in common}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for a,b in stable:union(a,b)
    roots={};out={}
    for t in sorted(common):
        r=find(t);roots.setdefault(r,len(roots));out[t]=roots[r]
    return out


def raw_mapping(data: List[Step]) -> Dict[int,int]:
    toks=sorted(set(t for _,b in data for t in b));return {t:i for i,t in enumerate(toks)}


def fit_transition(data: List[Step], mapping: Dict[int,int], conditioned=True):
    trans=defaultdict(Counter); prior=Counter()
    for i in range(len(data)-1):
        a,cur=data[i];_,nxt=data[i+1]
        cc=[mapping[t] for t in cur if t in mapping]; nn=[mapping[t] for t in nxt if t in mapping]
        if not cc or not nn:continue
        cs=Counter(cc).most_common(1)[0][0];ns=Counter(nn).most_common(1)[0][0]
        trans[(cs,a if conditioned else None)][ns]+=1;prior[ns]+=1
    return trans,prior


def predict_accuracy(train,test,mapping,conditioned=True):
    trans,prior=fit_transition(train,mapping,conditioned)
    correct=tot=0
    for i in range(len(test)-1):
        a,cur=test[i];_,nxt=test[i+1]
        cc=[mapping[t] for t in cur if t in mapping];nn=[mapping[t] for t in nxt if t in mapping]
        if not cc or not nn:continue
        cs=Counter(cc).most_common(1)[0][0];truth=Counter(nn).most_common(1)[0][0]
        cnt=trans.get((cs,a if conditioned else None))
        pred=(cnt.most_common(1)[0][0] if cnt else prior.most_common(1)[0][0])
        correct+=pred==truth;tot+=1
    return correct/max(1,tot)


def cluster_stability(data):
    mid=len(data)//2;m1=mdl_mapping(data[:mid]);m2=mdl_mapping(data[mid:])
    common=sorted(set(m1)&set(m2)); agree=tot=0
    for i,a in enumerate(common):
        for b in common[i+1:]:
            agree += ((m1[a]==m1[b])==(m2[a]==m2[b]));tot+=1
    return agree/max(1,tot)


def dominant_code(burst, mapping):
    vals=[mapping[t] for t in burst if t in mapping]
    return Counter(vals).most_common(1)[0][0] if vals else None


def choose_action_from_model(trans, mapping, current_burst, goal_burst, actions, horizon=3):
    cs=dominant_code(current_burst,mapping); gs=dominant_code(goal_burst,mapping)
    if cs is None or gs is None:return actions[0]
    best=None
    for h in range(1,horizon+1):
      for seq in itertools.product(actions, repeat=h):
        dist={cs:1.0}
        for a in seq:
            nd=defaultdict(float)
            for state,p0 in dist.items():
                cnt=trans.get((state,a),Counter());tot=sum(cnt.values())
                if not tot:
                    nd[state]+=p0
                else:
                    for nxt,v in cnt.items():nd[nxt]+=p0*(v/tot)
            dist=dict(nd)
        score=dist.get(gs,0.0)
        cand=(score,-h,tuple(-x for x in seq),seq[0])
        if best is None or cand>best:best=cand
    return best[3]


def control_score(world, train, mapping, trials=120, seed=0, max_steps=4):
    rr=random.Random(seed); wins=0
    trans,_=fit_transition(train,mapping,True)
    for _ in range(trials):
        current=rr.randrange(world.spec.states); goal=rr.randrange(world.spec.states)
        while goal==current:goal=rr.randrange(world.spec.states)
        world.state=goal;goal_obs=world.burst()
        world.state=current;cur=world.burst()
        success=False
        for _step in range(max_steps):
            action=choose_action_from_model(trans,mapping,cur,goal_obs,world.action_tokens,3)
            world.state=world.transition[(world.state,action)]
            cur=world.burst()
            if world.state==goal:
                success=True;break
        wins+=success
    return wins/trials


def run_seed(seed:int):
    w=AnonymousEventWorld(seed)
    stream=w.raw_rollout(1400,seed+90000)
    data=reconstruct_steps(stream,w.action_tokens)
    train=data[:1000];test=data[1000:]
    absmap=independent_mapping(train); raw=raw_mapping(train)
    pa=predict_accuracy(train,test,absmap,True)
    pr=predict_accuracy(train,test,raw,True)
    pn=predict_accuracy(train,test,absmap,False)
    rr=random.Random(seed+12345); actions=[a for a,_ in train];rr.shuffle(actions)
    shuffled=[(actions[i],b) for i,(_,b) in enumerate(train)]
    ps=predict_accuracy(shuffled,test,absmap,True)
    stab=cluster_stability(train)
    ca=control_score(w,train,absmap,120,seed+777)
    cr=control_score(w,train,raw,120,seed+1777)
    return dict(pred_abs=pa,pred_raw=pr,pred_no_action=pn,pred_shuffle=ps,control_abs=ca,control_raw=cr,stability=stab,clusters=len(set(absmap.values())),tokens=len(absmap))

GATES={
    "prediction_gain_over_raw":0.08,
    "action_ablation_gap":0.08,
    "shuffle_gap":0.08,
    "stability":0.85,
    "control_gain_over_raw":0.15,
    "control_absolute":0.70,
}

def benchmark(seeds=80):
    rows=[run_seed(416000+s) for s in range(seeds)]
    mean=lambda k:statistics.mean(r[k] for r in rows)
    summary={k:mean(k) for k in rows[0]}
    summary.update({
      "seeds":seeds,
      "prediction_gain_over_raw":mean("pred_abs")-mean("pred_raw"),
      "action_ablation_gap":mean("pred_abs")-mean("pred_no_action"),
      "shuffle_gap":mean("pred_abs")-mean("pred_shuffle"),
      "control_gain_over_raw":mean("control_abs")-mean("control_raw"),
    })
    summary["gate_pass"]=(summary["prediction_gain_over_raw"]>=GATES["prediction_gain_over_raw"] and
      summary["action_ablation_gap"]>=GATES["action_ablation_gap"] and
      summary["shuffle_gap"]>=GATES["shuffle_gap"] and summary["stability"]>=GATES["stability"] and
      summary["control_gain_over_raw"]>=GATES["control_gain_over_raw"] and summary["control_abs"]>=GATES["control_absolute"])
    return summary,rows

if __name__=='__main__':
    import json
    s,_=benchmark();print(json.dumps(s,indent=2))

class SelfOrganizingLearner:
    """Automatic v4.16 learner.

    Public learner inputs are deliberately narrow:
      * one timestamped anonymous token stream;
      * a set of opaque tokens the learner itself is able to emit as interventions.

    It is not given world states, entity IDs, Boolean values, operator names, alias groups,
    transition tables, or hidden benchmark semantics.
    """
    def __init__(self, intervention_tokens: Iterable[int]):
        self.intervention_tokens = tuple(intervention_tokens)
        self.steps: List[Step] = []
        self.mapping: Dict[int,int] = {}
        self.transition = None
        self.prior = None

    def fit(self, stream: List[Event]) -> "SelfOrganizingLearner":
        self.steps = reconstruct_steps(stream, self.intervention_tokens)
        if len(self.steps) < 20:
            raise ValueError("insufficient event history for independent rediscovery")
        self.mapping = independent_mapping(self.steps)
        self.transition, self.prior = fit_transition(self.steps, self.mapping, True)
        return self

    def represent(self, event_tokens: Iterable[int]) -> List[int]:
        return [self.mapping[t] for t in event_tokens if t in self.mapping]

    def choose_intervention(self, current_events: List[int], goal_events: List[int], horizon: int = 3) -> int:
        if self.transition is None:
            raise RuntimeError("fit() must be called first")
        return choose_action_from_model(self.transition, self.mapping, current_events, goal_events,
                                        self.intervention_tokens, horizon)

    def discovered_groups(self) -> List[List[int]]:
        groups=defaultdict(list)
        for t,c in self.mapping.items():groups[c].append(t)
        return [sorted(v) for _,v in sorted(groups.items())]
