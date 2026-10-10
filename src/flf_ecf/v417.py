from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math, random, statistics
from typing import Dict, List, Sequence, Tuple

Vector = Tuple[float, ...]

@dataclass
class ContinuousSegment:
    sensors: List[Vector]
    actuators: List[Vector]
    reward: float
    hidden_state: int | None = None

class ContinuousAnonymousWorld:
    """Synthetic continuous controlled process.

    Hidden states and actuator sectors are benchmark internals only.  The learner
    receives continuous sensor vectors, continuous actuator vectors and scalar reward.
    """
    def __init__(self, seed: int, states: int = 6, sensor_dim: int = 5,
                 actuator_dim: int = 2, noise: float = 0.06):
        self.r = random.Random(seed)
        self.S, self.sd, self.ad, self.noise = states, sensor_dim, actuator_dim, noise
        self.mus = [
            [1.5*math.sin((s+1)*(j+1)*0.91) + 1.2*math.cos((s+2)*(j+1)*0.47)
             for j in range(sensor_dim)]
            for s in range(states)
        ]
        rot = self.r.random()*2*math.pi
        self._dirs = [(math.cos(a+rot), math.sin(a+rot))
                      for a in (0.0, 2*math.pi/3, 4*math.pi/3)]
        self._effect = (-1, 1, 2)
        self._preferred = [(2*s+1) % 3 for s in range(states)]
        self.state = self.r.randrange(states)

    def _effect_id(self, u: Vector) -> int:
        return max(range(3), key=lambda i: u[0]*self._dirs[i][0] + u[1]*self._dirs[i][1])

    def observe(self, duration: int | None = None) -> Tuple[List[Vector], int]:
        if duration is None:
            duration = self.r.randint(3, 7)
        s = self.state
        out = []
        for t in range(duration):
            frac = (t-(duration-1)/2)/max(1, duration)
            out.append(tuple(
                self.mus[s][j] + 0.025*frac*(j+1) + self.r.gauss(0, self.noise)
                for j in range(self.sd)
            ))
        return out, s

    def advance(self, u: Vector) -> float:
        s = self.state
        e = self._effect_id(u)
        p = 0.86 if e == self._preferred[s] else 0.14
        reward = float(self.r.random() < p)
        self.state = (self.state + self._effect[e]) % self.S
        return reward

    def make_segment(self, u: Vector, duration: int | None = None) -> ContinuousSegment:
        sensors, s = self.observe(duration)
        actuators = [tuple(u[k] + self.r.gauss(0, 0.04) for k in range(self.ad))
                     for _ in range(len(sensors))]
        avg = tuple(statistics.mean(a[k] for a in actuators) for k in range(self.ad))
        reward = self.advance(avg)
        return ContinuousSegment(sensors, actuators, reward, s)

    def rollout(self, n_segments: int, policy_seed: int) -> List[ContinuousSegment]:
        rr = random.Random(policy_seed)
        out = []
        for _ in range(n_segments):
            ang = rr.random()*2*math.pi
            mag = 0.7 + 0.8*rr.random()
            out.append(self.make_segment((mag*math.cos(ang), mag*math.sin(ang))))
        return out

def flatten_segments(segs: Sequence[ContinuousSegment]):
    xs: List[Vector] = []; us: List[Vector] = []; rs: List[float] = []
    ends: List[int] = []; hidden: List[int | None] = []
    for seg in segs:
        for t, (x, u) in enumerate(zip(seg.sensors, seg.actuators)):
            xs.append(x); us.append(u)
            rs.append(seg.reward if t == len(seg.sensors)-1 else 0.0)
            hidden.append(seg.hidden_state)
            if t == len(seg.sensors)-1:
                ends.append(len(xs)-1)
    return xs, us, rs, ends, hidden

def _mean_std(xs: Sequence[Vector]):
    d = len(xs[0])
    m = [statistics.mean(x[j] for x in xs) for j in range(d)]
    sd = [statistics.pstdev(x[j] for x in xs) or 1.0 for j in range(d)]
    return m, sd

def _normalize(xs: Sequence[Vector], m, sd):
    return [tuple((x[j]-m[j])/sd[j] for j in range(len(m))) for x in xs]

def _dist(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def _unit(u: Vector) -> Vector:
    z = math.sqrt(sum(x*x for x in u)) or 1.0
    return tuple(x/z for x in u)

def discover_change_threshold(zs: Sequence[Vector]) -> float:
    """Data-driven threshold: largest relative gap in standardized step distances."""
    ds = [_dist(zs[i], zs[i-1]) for i in range(1, len(zs))]
    s = sorted(ds)
    lo = max(1, int(0.05*len(s)))
    hi = max(lo+1, int(0.97*len(s)))
    i = max(range(lo, hi), key=lambda q: (s[q+1]-s[q])/(s[q]+1e-9))
    return (s[i]+s[i+1])/2

def segment_stream(xs, us, rs, m=None, sd=None, threshold=None):
    if m is None:
        m, sd = _mean_std(xs)
    zs = _normalize(xs, m, sd)
    if threshold is None:
        threshold = discover_change_threshold(zs)
    bounds = [0]
    for i in range(1, len(zs)):
        if _dist(zs[i], zs[i-1]) > threshold:
            bounds.append(i)
    bounds.append(len(zs))
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        emb = tuple(statistics.mean(zs[t][j] for t in range(a, b)) for j in range(len(m)))
        act = tuple(statistics.mean(us[t][j] for t in range(a, b)) for j in range(len(us[0])))
        segs.append((emb, act, sum(rs[a:b]), a, b))
    return segs, (m, sd, threshold)

def _kmeans(data: Sequence[Vector], k: int, seed: int, iters: int = 50):
    rr = random.Random(seed)
    n, d = len(data), len(data[0])
    cent = [data[rr.randrange(n)]]
    while len(cent) < k:
        i = max(range(n), key=lambda q: min(_dist(data[q], c) for c in cent))
        cent.append(data[i])
    labels = [-1]*n
    for _ in range(iters):
        new = [min(range(k), key=lambda j: _dist(x, cent[j])) for x in data]
        if new == labels:
            break
        labels = new
        nc = []
        for j in range(k):
            pts = [data[i] for i in range(n) if labels[i] == j]
            nc.append(cent[j] if not pts else tuple(statistics.mean(p[z] for p in pts) for z in range(d)))
        cent = nc
    sse = sum(_dist(data[i], cent[labels[i]])**2 for i in range(n))
    return labels, cent, sse

def choose_state_count(data: Sequence[Vector], seed: int, max_k: int = 10):
    """Generic elbow discovery; hidden state count is never supplied."""
    vals = []
    for k in range(1, min(max_k, len(data)-1)+1):
        lab, c, sse = _kmeans(data, k, seed+k)
        vals.append((k, sse, lab, c))
    logs = [math.log(max(v[1], 1e-12)) for v in vals]
    curv = []
    for i in range(1, len(vals)-1):
        curv.append(((logs[i-1]-logs[i]) - (logs[i]-logs[i+1]), i))
    return vals[max(curv)[1]]

def _entropy(vals: Sequence[int]) -> float:
    c = Counter(vals); n = len(vals)
    return -sum((v/n)*math.log(max(v/n, 1e-12)) for v in c.values())

def _conditional_impurity(examples) -> float:
    by = defaultdict(list)
    for u, s, nx in examples:
        by[s].append(nx)
    return sum(len(v)*_entropy(v) for v in by.values())

@dataclass
class _ActionNode:
    idxs: List[int]
    dim: int | None = None
    threshold: float | None = None
    left: "_ActionNode | None" = None
    right: "_ActionNode | None" = None
    leaf: int | None = None

class ActuatorEquivalenceField:
    """Effect-grounded partition of a continuous actuator space.

    No action-class count is supplied. Regions are split only when they compress
    action-conditioned transition uncertainty enough to pay an MDL-like penalty.
    """
    def __init__(self, min_leaf: int = 8, penalty: float = 2.0, max_depth: int = 8):
        self.min_leaf, self.penalty, self.max_depth = min_leaf, penalty, max_depth

    def fit(self, actions, states, next_states):
        self.examples = [(_unit(u), s, nx) for u, s, nx in zip(actions, states, next_states)]
        self.leaves: List[_ActionNode] = []
        self.root = self._build(list(range(len(self.examples))), 0)
        for i, node in enumerate(self.leaves):
            node.leaf = i
        self.prototypes = []
        for node in self.leaves:
            pts = [actions[i] for i in node.idxs]
            probe = pts[::max(1, len(pts)//20)]
            self.prototypes.append(min(pts, key=lambda u: sum(_dist(_unit(u), _unit(v)) for v in probe)))
        return self

    def _build(self, idxs, depth):
        node = _ActionNode(idxs)
        ex = [self.examples[i] for i in idxs]
        parent = _conditional_impurity(ex)
        if depth >= self.max_depth or len(idxs) < 2*self.min_leaf:
            self.leaves.append(node); return node
        best = None
        d = len(ex[0][0])
        for j in range(d):
            vals = sorted(set(round(self.examples[i][0][j], 6) for i in idxs))
            if len(vals) < 4:
                continue
            for q in (0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85):
                p = min(len(vals)-2, max(1, int(q*(len(vals)-1))))
                th = (vals[p]+vals[p+1])/2
                L = [i for i in idxs if self.examples[i][0][j] <= th]
                R = [i for i in idxs if self.examples[i][0][j] > th]
                if min(len(L), len(R)) < self.min_leaf:
                    continue
                gain = parent - _conditional_impurity([self.examples[i] for i in L]) \
                              - _conditional_impurity([self.examples[i] for i in R])
                if best is None or gain > best[0]:
                    best = (gain, j, th, L, R)
        if best is None or best[0] <= self.penalty*math.log(len(idxs)+1):
            self.leaves.append(node); return node
        _, j, th, L, R = best
        node.dim, node.threshold = j, th
        node.left = self._build(L, depth+1)
        node.right = self._build(R, depth+1)
        return node

    def assign(self, u: Vector) -> int:
        x = _unit(u); n = self.root
        while n.leaf is None:
            n = n.left if x[n.dim] <= n.threshold else n.right
        return n.leaf

def _random_segments(xs, us, rs, nseg, seed, m, sd):
    rr = random.Random(seed)
    zs = _normalize(xs, m, sd)
    cuts = sorted(rr.sample(range(1, len(xs)), min(nseg-1, len(xs)-1)))
    bounds = [0] + cuts + [len(xs)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        emb = tuple(statistics.mean(zs[t][j] for t in range(a,b)) for j in range(len(m)))
        act = tuple(statistics.mean(us[t][j] for t in range(a,b)) for j in range(len(us[0])))
        out.append((emb, act, sum(rs[a:b]), a, b))
    return out

class ContinuousOperationalField:
    """v4.17 learner: continuous stream -> boundaries -> states -> actuator equivalence."""
    def __init__(self, seed: int = 0, force_k: int | None = None,
                 ignore_actuators: bool = False, random_segmentation: bool = False):
        self.seed, self.force_k = seed, force_k
        self.ignore_actuators, self.random_segmentation = ignore_actuators, random_segmentation

    def fit_stream(self, xs, us, rs):
        self.m, self.sd = _mean_std(xs)
        auto, pars = segment_stream(xs, us, rs, self.m, self.sd)
        self.threshold = pars[2]
        segs = _random_segments(xs, us, rs, len(auto), self.seed, self.m, self.sd) \
               if self.random_segmentation else auto
        return self._fit_segments(segs)

    def _fit_segments(self, segs):
        data = [s[0] for s in segs]
        if self.force_k is not None:
            lab, c, _ = _kmeans(data, self.force_k, self.seed+11)
            self.K = self.force_k
        else:
            self.K, _, lab, c = choose_state_count(data, self.seed+11)
        self.centroids, self.labels, self.train_segments = c, lab, segs
        actions = [s[1] for s in segs[:-1]]
        states, nexts = lab[:-1], lab[1:]
        if self.ignore_actuators:
            self.action_field = None
            self.A = 1
            alabs = [0]*len(actions)
        else:
            self.action_field = ActuatorEquivalenceField().fit(actions, states, nexts)
            self.A = len(self.action_field.leaves)
            alabs = [self.action_field.assign(a) for a in actions]
        self.tcounts = [[[0]*self.K for _ in range(self.A)] for __ in range(self.K)]
        self.tot = [[0]*self.A for _ in range(self.K)]
        self.rsum = [[0.0]*self.A for _ in range(self.K)]
        self.rn = [[0]*self.A for _ in range(self.K)]
        self.by_state_actions = defaultdict(list)
        for i, (s, nx, a) in enumerate(zip(states, nexts, alabs)):
            self.tcounts[s][a][nx] += 1; self.tot[s][a] += 1
            self.rsum[s][a] += segs[i][2]; self.rn[s][a] += 1
            self.by_state_actions[s].append((_unit(segs[i][1]), nx))
        self.action_prototypes = [(0.0,)*len(segs[0][1])] if self.action_field is None \
                                 else self.action_field.prototypes
        return self

    def segment(self, xs, us, rs):
        return segment_stream(xs, us, rs, self.m, self.sd, self.threshold)[0]

    def state_of(self, emb: Vector) -> int:
        return min(range(self.K), key=lambda j: _dist(emb, self.centroids[j]))

    def transition_probs(self, s: int, a: int):
        aa = 0 if self.ignore_actuators else a
        n = self.tot[s][aa]; alpha = 0.3
        return [(self.tcounts[s][aa][j]+alpha)/(n+alpha*self.K) for j in range(self.K)]

    def reward_estimate(self, s: int, a: int) -> float:
        aa = 0 if self.ignore_actuators else a
        return (self.rsum[s][aa]+0.5)/(self.rn[s][aa]+1.0)

    def predict_embedding(self, emb: Vector, u: Vector) -> Vector:
        s = self.state_of(emb)
        if self.ignore_actuators or not self.by_state_actions[s]:
            p = self.transition_probs(s, 0)
            return tuple(sum(p[j]*self.centroids[j][d] for j in range(self.K))
                         for d in range(len(emb)))
        uu = _unit(u)
        _, nx = min(self.by_state_actions[s], key=lambda row: _dist(uu, row[0]))
        return self.centroids[nx]

    def choose_action(self, emb: Vector) -> Vector:
        s = self.state_of(emb)
        a = max(range(self.A), key=lambda j: self.reward_estimate(s, j))
        return self.action_prototypes[a]

    def z_to_raw(self, z: Vector) -> Vector:
        return tuple(z[j]*self.sd[j] + self.m[j] for j in range(len(z)))

class RawContinuousKNN:
    """Strong no-state-abstraction diagnostic baseline."""
    def __init__(self, k: int = 10):
        self.k = k

    def fit_segments(self, segs):
        self.rows = []
        for i in range(len(segs)-1):
            self.rows.append((segs[i][0], _unit(segs[i][1]), segs[i+1][0], segs[i][2]))
        return self

    def predict_embedding(self, emb: Vector, u: Vector) -> Vector:
        uu = _unit(u)
        scored = sorted(((_dist(emb,x)**2 + 0.7*_dist(uu,a)**2, nx)
                         for x,a,nx,r in self.rows), key=lambda z:z[0])[:self.k]
        w = [1/(d+1e-4) for d,_ in scored]; z = sum(w)
        return tuple(sum(wi*nx[j] for wi,(_,nx) in zip(w, scored))/z
                     for j in range(len(emb)))

def prediction_rmse(model, segs) -> float:
    out = []
    for i in range(len(segs)-1):
        p = model.predict_embedding(segs[i][0], segs[i][1])
        y = segs[i+1][0]
        out.append(math.sqrt(sum((a-b)**2 for a,b in zip(p,y))/len(y)))
    return statistics.mean(out)

def persistence_rmse(segs) -> float:
    return statistics.mean(
        math.sqrt(sum((a-b)**2 for a,b in zip(segs[i][0],segs[i+1][0]))/len(segs[i][0]))
        for i in range(len(segs)-1)
    )

def boundary_f1(true_ends, segs) -> float:
    last = segs[-1][4]-1
    true = {x for x in true_ends if x < last}
    pred = {s[4]-1 for s in segs[:-1]}
    tp = len(true & pred)
    p = tp/max(1,len(pred)); r = tp/max(1,len(true))
    return 2*p*r/max(1e-12,p+r)

def transform_sensors(xs, seed):
    rr = random.Random(seed); d = len(xs[0])
    perm = list(range(d)); rr.shuffle(perm)
    scales = [rr.choice((-1,1))*(0.5+2*rr.random()) for _ in range(d)]
    shifts = [-3+6*rr.random() for _ in range(d)]
    return [tuple(scales[j]*x[perm[j]]+shifts[j] for j in range(d)) for x in xs]

def random_control(seed: int, n_segments: int = 350) -> float:
    w = ContinuousAnonymousWorld(seed); rr = random.Random(90000+seed); total = 0.0
    for _ in range(n_segments):
        w.observe()
        ang = rr.random()*2*math.pi
        total += w.advance((math.cos(ang), math.sin(ang)))
    return total/n_segments

def representation_control(seed: int, model: ContinuousOperationalField,
                           n_segments: int = 350) -> float:
    w = ContinuousAnonymousWorld(seed); total = 0.0
    for _ in range(n_segments):
        sensors, _ = w.observe()
        emb = tuple((statistics.mean(x[j] for x in sensors)-model.m[j])/model.sd[j]
                    for j in range(len(model.m)))
        total += w.advance(model.choose_action(emb))
    return total/n_segments

def predictive_rediscovery(seed, xs, us, rs, probe_x, probe_u, probe_r) -> float:
    mid = len(xs)//2
    a = ContinuousOperationalField(seed+101).fit_stream(xs[:mid], us[:mid], rs[:mid])
    b = ContinuousOperationalField(seed+202).fit_stream(xs[mid:], us[mid:], rs[mid:])
    sa = a.segment(probe_x, probe_u, probe_r); sb = b.segment(probe_x, probe_u, probe_r)
    n = min(len(sa), len(sb))-1
    if n <= 0:
        return 0.0
    _, scale = _mean_std(xs)
    ds = []
    for i in range(n):
        pa = a.z_to_raw(a.predict_embedding(sa[i][0], sa[i][1]))
        pb = b.z_to_raw(b.predict_embedding(sb[i][0], sb[i][1]))
        ds.append(math.sqrt(sum(((pa[j]-pb[j])/scale[j])**2 for j in range(len(scale)))/len(scale)))
    return math.exp(-statistics.mean(ds))

def run_seed(seed: int, train_segments: int = 500, test_segments: int = 220):
    w = ContinuousAnonymousWorld(seed)
    tr = w.rollout(train_segments, 10000+seed)
    xs, us, rs, ends, _ = flatten_segments(tr)
    model = ContinuousOperationalField(seed).fit_stream(xs, us, rs)
    raw = RawContinuousKNN().fit_segments(model.train_segments)

    wt = ContinuousAnonymousWorld(seed)
    te = wt.rollout(test_segments, 20000+seed)
    tx, tu, trw, tends, _ = flatten_segments(te)
    tseg = model.segment(tx, tu, trw)

    no_action = ContinuousOperationalField(seed, ignore_actuators=True).fit_stream(xs, us, rs)
    shuffled_u = list(us); random.Random(30000+seed).shuffle(shuffled_u)
    shuffled = ContinuousOperationalField(seed).fit_stream(xs, shuffled_u, rs)
    random_seg = ContinuousOperationalField(seed, random_segmentation=True).fit_stream(xs, us, rs)
    fixed = ContinuousOperationalField(seed, force_k=4).fit_stream(xs, us, rs)

    txf = transform_sensors(tx, 40000+seed)
    xsf = transform_sensors(xs, 40000+seed)
    invariant = ContinuousOperationalField(seed).fit_stream(xsf, us, rs)

    return {
        "seed": seed,
        "segments": len(model.train_segments),
        "boundary_f1": boundary_f1(ends, model.train_segments),
        "states": model.K,
        "actuator_regions": model.A,
        "prediction_rmse": prediction_rmse(model, tseg),
        "persistence_rmse": persistence_rmse(tseg),
        "raw_knn_rmse": prediction_rmse(raw, tseg),
        "no_intervention_rmse": prediction_rmse(no_action, no_action.segment(tx, tu, trw)),
        "shuffled_actuator_rmse": prediction_rmse(shuffled, shuffled.segment(tx, tu, trw)),
        "random_segmentation_rmse": prediction_rmse(random_seg, random_seg.segment(tx, tu, trw)),
        "fixed_k4_rmse": prediction_rmse(fixed, fixed.segment(tx, tu, trw)),
        "invariant_rmse": prediction_rmse(invariant, invariant.segment(txf, tu, trw)),
        "rediscovery_similarity": predictive_rediscovery(seed, xs, us, rs, tx, tu, trw),
        "control_reward": representation_control(seed, model),
        "random_control_reward": random_control(seed),
    }

def benchmark(n_seeds: int = 10):
    rows = [run_seed(s) for s in range(n_seeds)]
    def mean(k): return statistics.mean(r[k] for r in rows)
    out = {
        "seeds": n_seeds,
        "mean_segments": mean("segments"),
        "boundary_f1": mean("boundary_f1"),
        "mean_states": mean("states"),
        "mean_actuator_regions": mean("actuator_regions"),
        "prediction_rmse": mean("prediction_rmse"),
        "persistence_rmse": mean("persistence_rmse"),
        "raw_knn_rmse": mean("raw_knn_rmse"),
        "no_intervention_rmse": mean("no_intervention_rmse"),
        "shuffled_actuator_rmse": mean("shuffled_actuator_rmse"),
        "random_segmentation_rmse": mean("random_segmentation_rmse"),
        "fixed_k4_rmse": mean("fixed_k4_rmse"),
        "invariant_rmse": mean("invariant_rmse"),
        "rediscovery_similarity": mean("rediscovery_similarity"),
        "control_reward": mean("control_reward"),
        "random_control_reward": mean("random_control_reward"),
    }
    out["prediction_gain_vs_raw_knn"] = (out["raw_knn_rmse"]-out["prediction_rmse"])/out["raw_knn_rmse"]
    out["shuffle_gap"] = (out["shuffled_actuator_rmse"]-out["prediction_rmse"])/out["prediction_rmse"]
    out["random_segmentation_gap"] = (out["random_segmentation_rmse"]-out["prediction_rmse"])/out["prediction_rmse"]
    out["no_intervention_gap"] = (out["no_intervention_rmse"]-out["prediction_rmse"])/out["prediction_rmse"]
    out["fixed_k4_gap"] = (out["fixed_k4_rmse"]-out["prediction_rmse"])/out["prediction_rmse"]
    out["invariance_relative_error"] = abs(out["invariant_rmse"]-out["prediction_rmse"])/out["prediction_rmse"]
    out["control_gain"] = out["control_reward"]-out["random_control_reward"]
    out["gate_pass"] = (
        out["boundary_f1"] >= 0.95 and
        out["prediction_gain_vs_raw_knn"] >= 0.20 and
        out["shuffle_gap"] >= 2.0 and
        out["random_segmentation_gap"] >= 2.0 and
        out["no_intervention_gap"] >= 2.0 and
        out["invariance_relative_error"] <= 0.02 and
        out["rediscovery_similarity"] >= 0.85 and
        out["control_reward"] >= 0.75 and
        out["control_gain"] >= 0.25
    )
    return out, rows
