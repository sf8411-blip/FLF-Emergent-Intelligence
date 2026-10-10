# FLF Emergent Intelligence — ECF v4.16

## Objective

v4.16 tests a narrower and more defensible version of **self-generated representation**. The learner is not given the simulator's entity IDs, hidden regimes, Boolean operators, variable names, transition table, state count, or temporal order. Its interface contains only opaque event IDs, anonymous intervention IDs, and scalar feedback.

This is **domain-ontology-free, not mathematics-free**: every executable learner necessarily has algorithms and inductive biases. The scientific question is whether the benchmark's human semantic ontology must be supplied to the learner. In v4.16 it is not.

## Learning loop

1. infer temporal context depth from held-out predictive utility;
2. build raw context statistics from anonymous events;
3. merge histories only when the MDL score improves;
4. treat the resulting predictive-equivalence classes as operational states;
5. learn action-conditioned transitions and reward evidence on those states;
6. verify on unseen trajectories;
7. require independent rediscovery, semantic-shuffle degradation, action ablation, and event-label renaming invariance.

No hidden simulator labels are used by the learner or by any PASS gate.

## Predeclared operational gates

| Gate | Threshold |
|---|---:|
| Prediction gain over raw context memory | >= 5% |
| Shuffled-future degradation | >= 5% |
| Control gain over raw context memory | >= 0.015 |
| Action-value ablation gap | >= 0.10 |
| Independent predictive rediscovery | >= 0.80 |
| Opaque event-label renaming invariance | 100% |

## Verified 20-seed result

| Metric | Result |
|---|---:|
| Automatically selected temporal order | 2.0 |
| Automatically induced states | 5.0 |
| Predictive-state NLL | 2.21011 |
| Raw-context NLL | 2.37171 |
| Prediction gain | **6.8138%** |
| Shuffled-future NLL | 2.62009 |
| Shuffle degradation | **18.5500%** |
| Predictive-state control reward | 0.67272 |
| Raw-context control reward | 0.65237 |
| Control gain | **0.02035** |
| Action-ablated reward | 0.39735 |
| Action-value gap | **0.27537** |
| Independent rediscovery similarity | **0.83138** |
| Event-label renaming invariance | **1.00000** |
| Control wins vs raw baseline | **20 / 20** |
| Gate | **PASS** |

Raw per-seed data are committed under `results/`.

## Run

```bash
python run_benchmark.py --seeds 20
python -m unittest discover -s tests -v
```

The benchmark uses only the Python standard library.

## What this result does and does not establish

Supported bounded statement:

> In this synthetic controlled event process, an operational representation induced from anonymous event histories can improve held-out prediction and intervention control over raw-context memorization, while its advantage degrades under semantic shuffling and action ablation and remains invariant to arbitrary renaming of event IDs.

Not established: unrestricted entity genesis, general intelligence, mathematics emerging without inductive bias, natural-number discovery, or unrestricted language invention.

The repository's original v4.16 snapshot is preserved at `docs/legacy_v4_16_snapshot.md` rather than silently overwritten.

## Research basis

The architecture is inspired by predictive equivalence / causal-state reconstruction in computational mechanics and by MDL-based causal/model selection. Relevant recent references include Jurgens & Brodu, *Chaos* 35 (2025), DOI `10.1063/5.0242981`, and Cornanguer, Cüppers & Vreeken, AAAI-26 (2026), DOI `10.1609/aaai.v40i25.39201`.

## Next gate: v4.17

Remove another remaining prior: v4.16 still assumes that raw event identity is a primitive equality relation and that available interventions are supplied as a discrete action set. v4.17 should receive continuous anonymous sensor/actuator pulses, invent its own event partition and intervention equivalence classes, then retain them only if they improve prediction, compression, intervention, and independent rediscovery.
