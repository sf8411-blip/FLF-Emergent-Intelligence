# FLF Emergent Intelligence — ECF v4.17

## Objective

v4.17 removes two more ontology assumptions from the learner interface: **discrete event identity** and a **discrete action class**. The learner receives only continuous anonymous sensor vectors, continuous anonymous actuator vectors, time order, and scalar feedback.

The benchmark simulator has hidden states and hidden actuator effects for evaluation only. These labels, their counts, and their semantics are never supplied to the learner.

This is **ontology-reduced, not mathematics-free**. An executable learner still requires algorithms and inductive biases. The scientific question is narrower: can useful event boundaries, predictive states, and intervention equivalence regions be induced from continuous experience without supplying the benchmark's human semantic classes?

## v4.17 learning loop

1. standardize continuous sensor channels from experience;
2. detect event boundaries from a data-driven gap in local change magnitude, without event labels;
3. summarize each discovered segment into an operational observation/action episode;
4. infer the number of predictive states from the distortion elbow rather than receiving a state count;
5. induce continuous actuator equivalence regions by recursively splitting actuator space only when action-conditioned transition uncertainty is reduced enough to pay a complexity penalty;
6. predict unseen trajectories using discovered predictive states plus continuous actuator similarity;
7. control the world using reward evidence attached to discovered states and actuator regions;
8. require semantic-shuffle degradation, random-segmentation degradation, no-intervention degradation, independent rediscovery, and sensor permutation/affine-rescaling invariance.

Hidden simulator labels are used only to score event-boundary F1 externally; they are not used by the learner.

## Predeclared v4.17 gates

| Gate | Threshold |
|---|---:|
| Event-boundary F1 | >= 0.95 |
| Prediction gain over raw continuous kNN | >= 20% |
| Shuffled-actuator degradation ratio | >= 2.0 |
| Random-segmentation degradation ratio | >= 2.0 |
| No-intervention degradation ratio | >= 2.0 |
| Sensor permutation/affine-rescaling relative error | <= 2% |
| Independent predictive rediscovery | >= 0.85 |
| Representation-based control reward | >= 0.75 |
| Control gain over random continuous control | >= 0.25 |

The thresholds are encoded in `src/flf_ecf/v417.py` and were not relaxed after the benchmark run.

## Verified 10-seed result

| Metric | Result |
|---|---:|
| Automatically discovered segments / run | 500.0 |
| Event-boundary F1 | **1.0000** |
| Automatically discovered predictive states | **6.0** |
| Automatically induced actuator regions | **6.7 mean** |
| v4.17 prediction RMSE | **0.04579** |
| Raw continuous kNN RMSE | 0.08814 |
| Prediction gain vs raw kNN | **48.05%** |
| Persistence RMSE | 1.46465 |
| No-intervention RMSE | 0.88513 |
| Shuffled-actuator RMSE | 0.99767 |
| Random-segmentation RMSE | 0.82120 |
| Fixed-k=4 RMSE | 0.70037 |
| Sensor permutation/affine-rescaling RMSE | **0.04579** |
| Independent rediscovery similarity | **0.92060** |
| Representation control reward | **0.86686** |
| Random continuous control reward | 0.39286 |
| Control gain | **0.47400** |
| Gate | **PASS** |

Raw per-seed data are committed at `results/v4.17-raw.json`; the aggregate result is at `results/v4.17-summary.json`.

## Run

```bash
python run_v417.py --seeds 10
python -m unittest discover -s tests -v
```

The benchmark uses only the Python standard library.

## Controls and interpretation

The PASS depends on operational structure, not merely smooth continuous data:

- shuffling actuator semantics raises prediction RMSE from 0.04579 to 0.99767;
- randomizing segmentation raises RMSE to 0.82120;
- removing intervention information raises RMSE to 0.88513;
- forcing the wrong state count (`k=4`) raises RMSE to 0.70037;
- arbitrary sensor-channel permutation plus affine rescaling leaves the normalized predictive error unchanged to numerical precision;
- two disjoint experience partitions independently recover predictive behavior with mean similarity 0.92060.

Supported bounded statement:

> In this synthetic continuous controlled process, a learner not given discrete event IDs, hidden state labels, a state count, or discrete action classes can induce useful temporal boundaries, predictive states, and actuator-effect regions that improve held-out prediction and intervention control over raw continuous baselines.

Not established: unrestricted entity genesis, unrestricted language invention, general intelligence, mathematics emerging without inductive bias, or transfer to natural physical sensor streams.

## Prior stage

The validated v4.16 anonymous-discrete-event implementation remains in `src/flf_ecf/v416.py`, with its original snapshot preserved under `docs/legacy_v4_16_snapshot.md`.

## Research context

Predictive State Representations provide a precedent for representing controlled-process state through predictions of future observables and have been generalized to continuous actions and observations. Recent work on dynamic tokenization likewise studies event-aligned segmentation of heterogeneous continuous sensing streams. v4.17 uses a deliberately simpler standard-library implementation so every benchmark decision remains inspectable.

## Next gate: v4.18

v4.17 still contains generic human-designed representation machinery: standardized distance, a change-gap segmentation rule, k-means-style state proposals, and an axis-aligned actuator partition procedure. A serious v4.18 should **make representation-building procedures themselves compete and evolve**. Candidate builders should be generated from generic local operations, retained only if they improve prediction + compression + intervention + independent rediscovery, and rejected when shuffled worlds obtain the same score.
