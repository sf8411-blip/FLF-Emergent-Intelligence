# FLF Emergent Intelligence — ECF v4.18

## Current objective
v4.18 tests **self-evolving representation builders**. Its genome controls temporal segmentation, feature construction, predictive-state cardinality, actuator partitioning, normalization, transition targets, policy form, intervention equivalence, and complexity pressure.

The benchmark learner receives continuous sensor vectors, continuous actuator vectors, temporal order, and scalar feedback. Hidden simulator state labels and hidden actuator classes are not supplied to the learner.

This is **not unrestricted meta-learning**. The genome language, mutation/crossover operators, clustering substrate, and benchmark world are still human-designed.

## Validated v4.18.5 result
The v4.18.5 snapshot uses five independent main evolutionary runs and three matched ablation runs. Historical thresholds and results remain in `docs/v4.18_research_log.md` and `results/v4.18-summary.json`.

## Bounded interpretation
Supported: in this synthetic family of continuous controlled worlds, an evolutionary population improved a supplied representation-builder genome over fixed and equal-budget random search under the documented benchmark.

Not established: general intelligence, unrestricted self-modifying code, unrestricted builder-language invention, mathematics emerging without inductive bias, or transfer to natural physical systems.

## Next gate: v4.19 — Builder-Language Genesis
v4.18 evolves builders **inside a supplied genome language**. v4.19 investigates whether persistent residual failures can motivate creation, validation, reuse, and transfer of a new builder operator.

A bounded prototype is on branch [`v4.19-builder-language-genesis`](https://github.com/sf8411-blip/FLF-Emergent-Intelligence/tree/v4.19-builder-language-genesis). It proposes whitelisted expression-tree compositions and promotes them only when held-out RMSE improves by a configurable threshold. The primitive vocabulary remains human-designed; this is operator-composition search, not unrestricted code generation.

See [`docs/v4.19_design.md`](docs/v4.19_design.md) for mechanism, limits, and acceptance gates.

## Run regression tests
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
The heavier v4.18 evolutionary benchmark is not run inside CI by default.
