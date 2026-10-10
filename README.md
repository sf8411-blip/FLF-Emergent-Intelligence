# FLF Emergent Intelligence — ECF v4.16

## Goal
v4.16 removes the Boolean/operator/pointwise ontology from the **learner interface**. The learner receives only:

1. a timestamped stream of opaque event tokens; and
2. opaque tokens it can physically emit as interventions.

It is **not** given entity IDs, hidden states, Boolean values, variable names, operator names, alias groups, or the world's transition table.

The learner builds an operational representation by combining:

- predictive equivalence of anonymous events;
- temporal co-occurrence;
- a prediction + reconstruction/description-length objective;
- action-conditioned transition evidence;
- independent rediscovery on disjoint experience partitions;
- short-horizon intervention planning over the discovered representation.

This is a synthetic operational test. The benchmark world is still human-designed, so a PASS is **not** proof that mathematics or ontology has emerged without priors. The narrower claim is that the tested learner is not handed the benchmark's hidden state/operator ontology.

## Predeclared gates

| Gate | Threshold |
|---|---:|
| Prediction gain over raw-token model | >= 0.08 |
| Action-ablation gap | >= 0.08 |
| Shuffled-action semantic gap | >= 0.08 |
| Independent-rediscovery stability | >= 0.85 |
| Multi-step control gain over raw | >= 0.15 |
| Absolute multi-step control success | >= 0.70 |

## Valid benchmark snapshot

The committed benchmark snapshot uses **30 seeds**. Larger attempted runs (50/80/100) exceeded the execution budget and are not treated as evidence.

| Metric | Result |
|---|---:|
| Abstract prediction accuracy | 0.93918 |
| Raw-token prediction accuracy | 0.19081 |
| Prediction gain | 0.74837 |
| No-action prediction | 0.32063 |
| Action-ablation gap | 0.61855 |
| Shuffled-action prediction | 0.34185 |
| Shuffle gap | 0.59733 |
| Abstract multi-step control | 0.96972 |
| Raw multi-step control | 0.80639 |
| Control gain | 0.16333 |
| Independent rediscovery stability | 0.93072 |
| Gate | **PASS** |

## Run

```bash
python run_benchmark.py --seeds 30 --raw-out raw.json
python -m unittest discover -s tests -v
```

## Interpretation
The result supports this bounded statement:

> An automatically induced operational representation from anonymous event streams can, in this synthetic world, improve held-out prediction and multi-step intervention control relative to raw-token memorization, while the advantage degrades under shuffled intervention semantics and action ablation.

It does **not** establish general intelligence, unrestricted entity genesis, unrestricted language invention, or mathematics emerging from a fully assumption-free world.

## Next gate (v4.17 candidate)
The next experiment should remove the remaining handcrafted representational bias in v4.16: fixed pairwise distance ingredients, a fixed threshold grid, and a known actuator-token set. A stronger system should generate and compete representation-building procedures themselves, retaining them only by predictive/compressive/interventional evidence and independent rediscovery.
