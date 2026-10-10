# FLF Emergent Intelligence — ECF v4.18

## Current objective

v4.18 tests **self-evolving representation builders**. Instead of committing to one fixed segmentation/state/action-abstraction procedure, the system evolves a genome whose fields control:

- temporal segmentation pressure;
- temporal context width;
- feature construction;
- predictive-state cardinality;
- actuator-equivalence cardinality;
- normalization;
- absolute-vs-delta transition targets;
- control-policy form;
- raw-action vs effect-conditioned intervention equivalence;
- internal complexity pressure.

The benchmark learner receives continuous sensor vectors, continuous actuator vectors, temporal order, and scalar feedback. Hidden simulator state labels and hidden actuator classes are never supplied to the learner.

This is **not unrestricted meta-learning**. The genome language, mutation/crossover operators, clustering substrate, and benchmark world are still human-designed. The narrower question is whether evolution can improve the *procedure that constructs an operational representation* and whether that advantage transfers across unseen synthetic worlds.

## Predeclared v4.18 gates

The following thresholds were fixed before the successful run and were not relaxed:

| Gate | Threshold |
|---|---:|
| Prediction gain vs fixed builder | >= 8% |
| Prediction gain vs equal-budget random builder search | >= 3% |
| Control gain vs fixed builder | >= 0.08 |
| Evolutionary gain over initial population | >= 0.03 |
| Mutation ablation gap | >= 0.005 |
| Crossover ablation gap | >= -0.005 |
| Shuffled-actuator degradation | >= 50% |
| Pure-noise degradation | >= 100% |
| Independent operational rediscovery | >= 0.80 |

## Validated v4.18.5 result

The valid snapshot uses **5 independent main evolutionary runs** and **3 matched ablation runs**. Evolution and random search receive equal builder-evaluation budgets in each main run.

| Metric | Result |
|---|---:|
| Held-out RMSE | **0.72455** |
| Fixed-builder RMSE | 2.01487 |
| Equal-budget random-search RMSE | 1.15725 |
| Prediction gain vs fixed | **64.04%** |
| Prediction gain vs random search | **37.39%** |
| Held-out control reward | **0.99833** |
| Fixed-builder control reward | 0.30417 |
| Control gain | **+0.69417** |
| Evolutionary gain | **+0.30807** |
| Mutation gap | **+0.39358** |
| Crossover gap | **+0.67986** |
| Shuffled-actuator degradation | **+256.17%** |
| Pure-noise degradation | **+123.72%** |
| Independent rediscovery | **0.80** |
| Gate | **PASS** |

Raw per-run data are committed in `results/v4.18-raw.json`; the aggregate result is in `results/v4.18-summary.json`. The complete sequence of failed screens, diagnosed bugs, and repairs is preserved in `docs/v4.18_research_log.md`.

## What changed before the PASS

v4.18 did not pass immediately. Several negative results were retained rather than hidden. The final result required three substantive repairs:

1. actuator equivalence could be induced from **observed intervention effects**, not only command geometry;
2. the online control loop was corrected to use the **post-intervention observation** rather than a one-step-stale state;
3. builder selection became rank-based and multi-objective over prediction, reward evidence, complexity, transfer stability, and control utility.

No gate threshold was reduced after seeing results.

## Run v4.18

```bash
python -m pip install -r requirements.txt
python run_v418.py --main-runs 5 --ablation-runs 3
python -m unittest discover -s tests -v
```

The full benchmark is intentionally not run inside CI because the evolutionary experiment is substantially heavier than the regression tests.

## Bounded interpretation

Supported statement:

> In this synthetic family of continuous controlled worlds, an evolutionary population can improve a supplied representation-builder genome over a fixed hand-coded builder and equal-budget random builder search, and the advantage depends on coherent intervention semantics and degrades strongly on pure-noise controls.

Not established: general intelligence, unrestricted self-modifying code, unrestricted builder-language invention, mathematics emerging without inductive bias, or transfer to natural physical systems.

## Prior validated stages

- `src/flf_ecf/v417.py`: v4.17 continuous anonymous sensor/actuator streams, automatic event/state/intervention abstraction.
- `src/flf_ecf/v416.py`: v4.16 anonymous discrete-event operational representation.
- `docs/legacy_v4_16_snapshot.md`: preserved historical v4.16 snapshot.

## Next gate: v4.19

v4.18 evolves builders **inside a supplied genome language**. The next unresolved prior is therefore builder-language genesis. A serious v4.19 should begin with a deliberately incomplete builder language and test whether persistent residual failures cause the system to create, validate, reuse, and transfer a genuinely new builder operator. New operators should be retained only when they improve compression, held-out prediction, intervention/control, transfer, falsification resistance, and independent rediscovery; shuffled/noise worlds must not produce the same additions.
