from __future__ import annotations
import math
import numpy as np
from . import v420_dev as v


def _search_1d(fn, lo: float, hi: float, rounds: int = 5, points: int = 9) -> float:
    """Deterministic bounded 1-D refinement over a continuous interval.

    The search grid is regenerated inside a shrinking interval each round; it is
    not the DSL constant grid and can return values anywhere in the interval.
    """
    a, b = float(lo), float(hi)
    best_x = 0.5 * (a + b)
    for _ in range(rounds):
        xs = np.linspace(a, b, points)
        scores = np.asarray([float(fn(float(x))) for x in xs])
        j = int(np.argmax(scores))
        best_x = float(xs[j])
        if j == 0:
            a, b = float(xs[0]), float(xs[1])
        elif j == len(xs) - 1:
            a, b = float(xs[-2]), float(xs[-1])
        else:
            a, b = float(xs[j - 1]), float(xs[j + 1])
    return best_x


def tune_constants_continuous(p: v.Program, env, residual, train_idx) -> v.Program:
    """Fit each learned constant by bounded 1-D search; charge remains in Program.mdl_cost."""
    tr = np.asarray(train_idx, int)
    res = np.asarray(residual, float)

    def score(q: v.Program) -> float:
        return abs(v.corr(v.eval_program(q, env)[tr], res[tr]))

    out = p
    # Coordinate refinement for two-constant rank programs; every coordinate is
    # fitted with the same 1-D search rule.
    for _ in range(2 if p.q is not None else 1):
        if out.p is not None:
            if out.kind in ('moddiff', 'modraw'):
                lo, hi = 0.50, 1.60
            elif out.kind == 'rankprod':
                lo, hi = 4.20, 15.80
            else:
                lo, hi = 1.60, 9.80
            px = _search_1d(lambda x: score(v.replace(out, p=x)), lo, hi)
            out = v.replace(out, p=px)
        if out.q is not None:
            qx = _search_1d(lambda x: score(v.replace(out, q=x)), 0.42, 0.88)
            out = v.replace(out, q=qx)
    return out


def make_trace_with_meta(family: str, seed: int, meta: dict, n: int = 520,
                         action_scale: float = 1.6, difficulty: float = 1.0):
    """Independent trace from the same latent world parameters.

    New stochastic innovations/actions are drawn from ``seed`` while the source
    world's period/window/quantile in ``meta`` is held fixed. This is the R6
    intervention challenge: same mechanism, different action amplitude/trace.
    """
    r = np.random.default_rng(int(seed))
    u = r.normal(size=(n, 2)) * float(action_scale)
    x1 = np.zeros(n); x2 = np.zeros(n)
    for t in range(1, n):
        x1[t] = .75 * x1[t-1] + .32 * r.normal() + .15 * u[t-1, 1]
        x2[t] = .6 * x2[t-1] + .4 * r.normal()

    if family == 'circular':
        P = float(meta['period'])
        phase = np.zeros(n)
        drive = .16 * u[:, 0] + .07 * np.tanh(x1) + .11 * r.normal(size=n)
        for t in range(1, n): phase[t] = phase[t-1] + drive[t]
        x0 = np.mod(phase, P)
        h = np.mod(x0 - v.lag(x0), P)
        env = {'x0': x0, 'x1': x1, 'x2': x2, 'u0': u[:, 0], 'u1': u[:, 1]}
        rr = np.random.default_rng(int(seed) + 777)
        for j in range(4):
            z = np.zeros(n)
            for t in range(1, n): z[t] = .75 * z[t-1] + .35 * rr.normal()
            env[f'd{j}'] = z
        y = .35*x1 + .18*v.lag(x1) + .12*x2*v.lag(x1) + difficulty*1.4*h + .05*u[:,1] + .08*r.normal(size=n)
        return env, np.asarray(y), np.asarray(h), {'period': P}

    if family == 'rank':
        x0 = np.zeros(n)
        for t in range(1, n): x0[t] = .8*x0[t-1] + .3*r.normal() + .08*u[t-1,0]
        w = float(meta['window']); q = float(meta['quantile'])
        h = (x1 - v.lag(x1)) * (v.weighted_rank(x0, w) > q)
        env = {'x0': x0, 'x1': x1, 'u0': u[:,0], 'u1': u[:,1]}
        y = .35*x1 + .15*v.lag(x1) + difficulty*2.0*h + .06*u[:,0] + .08*r.normal(size=n)
        return env, np.asarray(y), np.asarray(h), {'window': w, 'quantile': q}

    if family == 'null':
        x0 = np.zeros(n)
        for t in range(1, n): x0[t] = .7*x0[t-1] + .35*r.normal() + .1*u[t-1,0]
        env = {'x0':x0,'x1':x1,'x2':x2,'u0':u[:,0],'u1':u[:,1]}
        h = np.zeros(n)
        y = .6*x1 + .25*v.lag(x1) - .22*x0*x2 + .20*v.movmean_discrete(x2,3) + .08*u[:,0] + .08*r.normal(size=n)
        return env, np.asarray(y), h, {}
    raise ValueError(family)


def challenge_same_world(program: v.Program, family: str, meta: dict, seed: int,
                         B: int = 999, action_scale: float = 1.6):
    env, y, h, meta2 = make_trace_with_meta(family, seed, meta, action_scale=action_scale)
    base = v.b0_and_oracle(env, y, h)
    z = v.eval_program(program, env)
    m = v.extension_fit(base, y, z)
    actual = base['b0_val_rmse'] - m['val_rmse']
    rr = np.random.default_rng(int(seed) + 404)
    null = []
    for _ in range(int(B)):
        zp = v.block_permute(z, rr, 20)
        null.append(base['b0_val_rmse'] - v.extension_fit(base, y, zp)['val_rmse'])
    p = (1 + sum(x >= actual for x in null)) / (B + 1)
    return float(p), float(actual), meta2


def admission_same_world(program: v.Program, z, fitted, base, y, family: str,
                         meta: dict, challenge_seed: int):
    r2 = v.novelty_r2(z, base['X'], base['tr'])
    gain = (base['b0_val_rmse'] - fitted['val_rmse']) / max(base['b0_val_rmse'], 1e-12)
    if r2 > v.NOVELTY_R2_MAX or gain < v.VALIDATION_GAIN_MIN:
        return False, {'novelty_r2': float(r2), 'validation_gain': float(gain), 'challenge_p': 1.0}
    p, cg, cmeta = challenge_same_world(program, family, meta, challenge_seed)
    ok = p <= v.CHALLENGE_P_MAX
    return bool(ok), {'novelty_r2': float(r2), 'validation_gain': float(gain),
                      'challenge_p': float(p), 'challenge_gain': float(cg),
                      'challenge_meta': cmeta, 'action_scale': 1.6}


def oracle_program(family: str, meta: dict) -> v.Program:
    if family == 'circular':
        return v.Program('moddiff', 'x0', p=float(meta['period']))
    if family == 'rank':
        return v.Program('rankprod', 'x0', b='x1', p=float(meta['window']), q=float(meta['quantile']))
    raise ValueError(family)
