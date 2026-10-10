from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import heapq, math, random, statistics
from typing import Dict, List, Sequence, Tuple

Event = int
Context = Tuple[Event, ...]

@dataclass(frozen=True)
class Step:
    previous: Event
    current: Event
    action: int
    reward: int
    next_event: Event

class AnonymousEventWorld:
    """Controlled event-process benchmark.

    Hidden regimes are simulator internals only. The learner receives opaque event IDs,
    anonymous actions, and scalar reward. No regime/entity/operator labels are exposed.
    """
    def __init__(self, seed: int, event_types: int = 14, hidden_regimes: int = 5, actions: int = 3):
        r = random.Random(seed)
        self.r, self.K, self.S, self.A = r, event_types, hidden_regimes, actions
        self.labels: List[int] = []
        while len(self.labels) < self.K:
            x = r.getrandbits(63)
            if x not in self.labels:
                self.labels.append(x)
        contexts = [(i, j) for i in range(self.K) for j in range(self.K)]
        r.shuffle(contexts)
        self._regime = {h: n % self.S for n, h in enumerate(contexts)}
        self._next = []
        for s in range(self.S):
            by_action = []
            for a in range(self.A):
                weights = [0.25 + 0.25 * r.random() for _ in range(self.K)]
                weights[(2 * s + a) % self.K] += 3.0
                weights[(3 * s + 2 * a + 1) % self.K] += 1.2
                z = sum(weights)
                by_action.append([x / z for x in weights])
            self._next.append(by_action)
        self._reward = []
        for s in range(self.S):
            best = (2 * s + 1) % self.A
            row = []
            for a in range(self.A):
                p = 0.12 + 0.08 * r.random()
                if a == best:
                    p += 0.52
                elif a == (best + 1) % self.A:
                    p += 0.15
                row.append(min(0.9, p))
            self._reward.append(row)
        self._prev, self._curr = r.randrange(self.K), r.randrange(self.K)

    def _sample(self, probabilities: Sequence[float]) -> int:
        u, c = self.r.random(), 0.0
        for i, p in enumerate(probabilities):
            c += p
            if u <= c:
                return i
        return len(probabilities) - 1

    def reset(self) -> Tuple[Event, Event]:
        self._prev, self._curr = self.r.randrange(self.K), self.r.randrange(self.K)
        return self.labels[self._prev], self.labels[self._curr]

    def step(self, action: int) -> Step:
        s = self._regime[(self._prev, self._curr)]
        nxt = self._sample(self._next[s][action])
        reward = int(self.r.random() < self._reward[s][action])
        obs = Step(self.labels[self._prev], self.labels[self._curr], action, reward, self.labels[nxt])
        self._prev, self._curr = self._curr, nxt
        return obs


def collect(world: AnonymousEventWorld, n: int, policy_seed: int) -> List[Step]:
    rr = random.Random(policy_seed)
    world.reset()
    return [world.step(rr.randrange(world.A)) for _ in range(n)]


def _events(steps: Sequence[Step]) -> List[Event]:
    return [steps[0].previous, steps[0].current] + [s.next_event for s in steps]


def records(steps: Sequence[Step], order: int):
    ev = _events(steps)
    for i, s in enumerate(steps):
        end = i + 2
        if end < order:
            continue
        yield tuple(ev[end-order:end]), s.action, s.reward, s.next_event

class _Stats:
    def __init__(self, actions: int, tokens: int):
        self.A, self.K = actions, tokens
        self.next = [[0] * tokens for _ in range(actions)]
        self.r1 = [0] * actions
        self.rn = [0] * actions
        self.n = 0
    def add(self, a: int, t: int, r: int):
        self.next[a][t] += 1; self.r1[a] += r; self.rn[a] += 1; self.n += 1
    def merged(self, other: '_Stats') -> '_Stats':
        z = _Stats(self.A, self.K); z.n = self.n + other.n
        for a in range(self.A):
            z.r1[a] = self.r1[a] + other.r1[a]; z.rn[a] = self.rn[a] + other.rn[a]
            z.next[a] = [x + y for x, y in zip(self.next[a], other.next[a])]
        return z

class PredictiveStateField:
    """Induce operational states by MDL merging of histories with equivalent futures."""
    def __init__(self, order: int, alpha: float = 0.5):
        self.order, self.alpha = order, alpha
    def fit(self, steps: Sequence[Step], shuffle_futures: bool = False, seed: int = 0, ignore_actions: bool = False):
        labels = sorted(set(_events(steps)))
        self.to_index = {x: i for i, x in enumerate(labels)}; self.K = len(labels)
        self.A = 1 + max(s.action for s in steps); self.ignore_actions = ignore_actions
        if shuffle_futures:
            rr = random.Random(seed); outcomes = [(s.next_event, s.reward) for s in steps]; rr.shuffle(outcomes)
            steps = [Step(s.previous, s.current, s.action, outcomes[i][1], outcomes[i][0]) for i, s in enumerate(steps)]
        model_A = 1 if ignore_actions else self.A
        by_context: Dict[Context, _Stats] = {}
        for h, a, r, nxt in records(steps, self.order):
            aa = 0 if ignore_actions else a
            if h not in by_context: by_context[h] = _Stats(model_A, self.K)
            by_context[h].add(aa, self.to_index[nxt], r)
        N = max(2, sum(st.n for st in by_context.values()))
        params_per_state = model_A * self.K
        penalty = 0.5 * params_per_state * math.log(N)
        def nll(st: _Stats) -> float:
            out = 0.0
            for a in range(model_A):
                c = st.next[a]; n = sum(c)
                if n:
                    out -= sum(x * math.log(x / n) for x in c if x)
                rn, r1 = st.rn[a], st.r1[a]
                if rn:
                    p = min(max(r1 / rn, 1e-9), 1 - 1e-9)
                    out -= r1 * math.log(p) + (rn - r1) * math.log(1 - p)
            return out
        clusters = {i: ([h], st) for i, (h, st) in enumerate(by_context.items())}
        score = {i: nll(st) + penalty for i, (_, st) in clusters.items()}
        version = {i: 0 for i in clusters}; heap = []
        def push(i: int, j: int):
            if i == j or i not in clusters or j not in clusters: return
            if i > j: i, j = j, i
            sm = clusters[i][1].merged(clusters[j][1])
            delta = (nll(sm) + penalty) - score[i] - score[j]
            heapq.heappush(heap, (delta, i, j, version[i], version[j]))
        ids = list(clusters)
        for ii, i in enumerate(ids):
            for j in ids[ii+1:]: push(i, j)
        while heap:
            delta, i, j, vi, vj = heapq.heappop(heap)
            if delta >= 0: break
            if i not in clusters or j not in clusters or version[i] != vi or version[j] != vj: continue
            sm = clusters[i][1].merged(clusters[j][1])
            clusters[i] = (clusters[i][0] + clusters[j][0], sm)
            score[i] = nll(sm) + penalty; version[i] += 1
            del clusters[j], score[j], version[j]
            for k in list(clusters):
                if k != i: push(i, k)
        self.context_to_state = {}; self.states = {}
        for new_id, (_, (hs, st)) in enumerate(clusters.items()):
            self.states[new_id] = st
            for h in hs: self.context_to_state[h] = new_id
        return self

    def state(self, h: Context): return self.context_to_state.get(tuple(h[-self.order:]))
    def next_probability(self, h: Context, a: int, nxt: Event) -> float:
        j = self.state(h)
        if j is None or nxt not in self.to_index: return 1 / self.K
        aa = 0 if self.ignore_actions else a; st = self.states[j]; counts = st.next[aa]; n = sum(counts)
        return (counts[self.to_index[nxt]] + self.alpha) / (n + self.alpha * self.K)
    def reward_probability(self, h: Context, a: int) -> float:
        j = self.state(h)
        if j is None: return 0.5
        aa = 0 if self.ignore_actions else a; st = self.states[j]
        return (st.r1[aa] + self.alpha) / (st.rn[aa] + 2 * self.alpha)
    def action(self, h: Context) -> int:
        return max(range(self.A), key=lambda a: self.reward_probability(h, a))

class AutoField:
    """Select temporal context depth from held-out predictive utility, then induce states."""
    def __init__(self, max_order: int = 4, alpha: float = 0.5): self.max_order, self.alpha = max_order, alpha
    def _raw_score(self, tr: Sequence[Step], val: Sequence[Step], L: int) -> float:
        labels=sorted(set(_events(tr)));toi={x:i for i,x in enumerate(labels)};K=len(labels);A=1+max(s.action for s in tr)
        d={}
        for h,a,r,nxt in records(tr,L):
            if h not in d:d[h]=_Stats(A,K)
            if nxt in toi:d[h].add(a,toi[nxt],r)
        vals=[]
        for h,a,r,nxt in records(val,L):
            st=d.get(h)
            if st is None or nxt not in toi:p=1/K;q=.5
            else:
                c=st.next[a];n=sum(c);p=(c[toi[nxt]]+self.alpha)/(n+self.alpha*K)
                q=(st.r1[a]+self.alpha)/(st.rn[a]+2*self.alpha)
            vals.append(-math.log(max(p,1e-12))+.35*(q-r)**2)
        return statistics.mean(vals)+0.001*L
    def fit(self, steps: Sequence[Step], shuffle_futures: bool = False, seed: int = 0, ignore_actions: bool = False):
        cut=int(.78*len(steps));tr,val=steps[:cut],steps[cut:]
        self.selection=[(self._raw_score(tr,val,L),L) for L in range(1,self.max_order+1)]
        self.order=min(self.selection)[1]
        self.model=PredictiveStateField(self.order,self.alpha).fit(steps,shuffle_futures,seed,ignore_actions)
        return self
    @property
    def states(self): return self.model.states
    def next_probability(self,h,a,nxt): return self.model.next_probability(tuple(h[-self.order:]),a,nxt)
    def reward_probability(self,h,a): return self.model.reward_probability(tuple(h[-self.order:]),a)
    def action(self,h): return self.model.action(tuple(h[-self.order:]))

class ContextMemory:
    """No-abstraction baseline: memorize each raw context separately."""
    def __init__(self, order: int, alpha: float = 0.5): self.order, self.alpha = order, alpha
    def fit(self, steps: Sequence[Step]):
        labels = sorted(set(_events(steps))); self.to_index = {x:i for i,x in enumerate(labels)}; self.K=len(labels);self.A=1+max(s.action for s in steps)
        self.d={}
        for h,a,r,nxt in records(steps,self.order):
            if h not in self.d:self.d[h]=_Stats(self.A,self.K)
            self.d[h].add(a,self.to_index[nxt],r)
        return self
    def next_probability(self,h,a,nxt):
        h=tuple(h[-self.order:])
        if h not in self.d or nxt not in self.to_index:return 1/self.K
        st=self.d[h];c=st.next[a];n=sum(c);return (c[self.to_index[nxt]]+self.alpha)/(n+self.alpha*self.K)
    def reward_probability(self,h,a):
        h=tuple(h[-self.order:]);st=self.d.get(h)
        if st is None:return .5
        return (st.r1[a]+self.alpha)/(st.rn[a]+2*self.alpha)
    def action(self,h):return max(range(self.A),key=lambda a:self.reward_probability(h,a))

def prediction_loss(model, steps: Sequence[Step], order: int) -> float:
    return statistics.mean(-math.log(max(model.next_probability(h,a,nxt),1e-12)) for h,a,r,nxt in records(steps,order))

def control_reward(world_seed:int, model, horizon:int=3000, event_types:int=14, hidden_regimes:int=5) -> float:
    w=AnonymousEventWorld(world_seed,event_types,hidden_regimes); hist=list(w.reset()); total=0
    for _ in range(horizon):
        a=model.action(hist);s=w.step(a);total+=s.reward;hist.append(s.next_event)
    return total/horizon

def predictive_rediscovery(steps:Sequence[Step], max_order:int=4) -> float:
    mid=len(steps)//2
    a=AutoField(max_order).fit(steps[:mid]); b=AutoField(max_order).fit(steps[mid:])
    probe=steps[-min(1800,len(steps)):]; sims=[]
    L=max(a.order,b.order)
    for h,act,r,nxt in records(probe,L):
        labels=sorted(set(a.model.to_index)&set(b.model.to_index))
        if not labels:continue
        tv=.5*sum(abs(a.next_probability(h,act,t)-b.next_probability(h,act,t)) for t in labels)
        sims.append(max(0.0,1.0-tv))
    return statistics.mean(sims) if sims else 0.0

def relabel_steps(steps:Sequence[Step],seed:int)->List[Step]:
    rr=random.Random(seed);labels=sorted(set(_events(steps)));perm=labels[:];rr.shuffle(perm);mp=dict(zip(labels,perm))
    return [Step(mp[s.previous],mp[s.current],s.action,s.reward,mp[s.next_event]) for s in steps]

def run_seed(seed:int,train_n:int=6000,test_n:int=4000,event_types:int=14,hidden_regimes:int=5):
    train=collect(AnonymousEventWorld(seed,event_types,hidden_regimes),train_n,100000+seed)
    test=collect(AnonymousEventWorld(seed,event_types,hidden_regimes),test_n,200000+seed)
    field=AutoField(4).fit(train); baseline=ContextMemory(field.order).fit(train)
    shuffled=AutoField(4).fit(train,shuffle_futures=True,seed=300000+seed)
    ablated=AutoField(4).fit(train,ignore_actions=True)
    nll=prediction_loss(field,test,field.order); base_nll=prediction_loss(baseline,test,field.order); shuf_nll=prediction_loss(shuffled,test,shuffled.order)
    rew=control_reward(seed,field,event_types=event_types,hidden_regimes=hidden_regimes)
    brew=control_reward(seed,baseline,event_types=event_types,hidden_regimes=hidden_regimes)
    arew=control_reward(seed,ablated,event_types=event_types,hidden_regimes=hidden_regimes)
    red=predictive_rediscovery(train)
    renamed_train=relabel_steps(train,500000+seed); renamed=AutoField(4).fit(renamed_train)
    inv=(renamed.order==field.order and len(renamed.states)==len(field.states))
    return {'seed':seed,'order':field.order,'states':len(field.states),'prediction_nll':nll,'baseline_nll':base_nll,'shuffled_nll':shuf_nll,
            'control_reward':rew,'baseline_reward':brew,'action_ablated_reward':arew,'rediscovery_similarity':red,'renaming_invariant':inv}

def benchmark(n_seeds:int=20):
    rows=[run_seed(s) for s in range(n_seeds)]; mean=lambda k:statistics.mean(r[k] for r in rows)
    out={'seeds':n_seeds,'mean_selected_order':mean('order'),'mean_states':mean('states'),'prediction_nll':mean('prediction_nll'),
         'baseline_nll':mean('baseline_nll'),'shuffled_nll':mean('shuffled_nll'),'control_reward':mean('control_reward'),
         'baseline_reward':mean('baseline_reward'),'action_ablated_reward':mean('action_ablated_reward'),
         'rediscovery_similarity':mean('rediscovery_similarity'),'renaming_invariance_rate':mean('renaming_invariant')}
    out['prediction_gain']=(out['baseline_nll']-out['prediction_nll'])/out['baseline_nll']
    out['shuffle_gap']=(out['shuffled_nll']-out['prediction_nll'])/out['prediction_nll']
    out['control_gain']=out['control_reward']-out['baseline_reward'];out['action_value_gap']=out['control_reward']-out['action_ablated_reward']
    out['gate_pass']=(out['prediction_gain']>=0.05 and out['shuffle_gap']>=0.05 and out['control_gain']>=0.015 and
                      out['action_value_gap']>=0.10 and out['rediscovery_similarity']>=0.80 and out['renaming_invariance_rate']==1.0)
    return out,rows
