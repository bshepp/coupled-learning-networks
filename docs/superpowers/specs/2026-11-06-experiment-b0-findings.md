# Findings — Experiment B0: does the mechanical rule *learn* a representable task? (elastic allostery)

**Date:** 2026-11-06
**Question.** Experiment B certified the mechanical two-phase update *per step* (keystone ρ → 1 as β → 0)
and foreclosed on parity-3, but never showed the rule *learning* anything at network scale on the elastic
medium — the positive control the commons' `network-foreclose-needs-a-positive-control` lesson asks for,
and the prerequisite for any read-noise (H5) study on this substrate. B0 runs the literature-canonical
friendly task — **mechanical allostery** (one input node displaced, a prescribed transfer ratio at a
distant output node; Rocks et al. 2017) — in three variants relative to the untrained lattice's own
ratio r₀: **attenuate** (0.5 r₀), **amplify** (2 r₀), **invert** (−0.5 r₀).
**Verdicts:** __VERDICTS__
**Audit:** headline runs 100 % `PHYSICS_LOCAL`; ceiling and baseline in separate `OFFLINE_GRADIENT` logs.

> **Observed** = measured; **Interpretation** separate. Pre-registered verdict: `tasks.verdict_b0`
> (multistability gate on the tail jump rate > 0.2 → representability via the converged ceiling > 0.1
> → PASS if local error ≤ max(1.15 × matched-budget GD, 1e-3) with init alignment sign ≥ 0.9, ρ ≥ 0.8 →
> FORECLOSE-LEARNABILITY if the rule does not even halve the untrained error → AMBIGUOUS). Task verdict
> = PASS only if every seed passes, else the most severe seed status.

---

## Observed

### Apparatus
- 4×4 triangulated lattice; input = bottom-middle node's y-displacement, amplitudes u ∈ 0.1·U[0.3, 1]
  (6 patterns); output = top-right node's y-displacement; bottom-left node pinned, bottom-right pinned in y.
- Untrained (uniform k = 1, unstressed) transfer ratio **r₀ = 0.5165** at d_max = 0.1 (0.526 at 0.05, 0.497 at
  0.2 — the response is linear to ~2 % at d_max = 0.1).
- Per cell (task × seed; seeds randomize k₀ ∈ U[0.5, 2], ℓ₀ ∈ ℓ·U[0.9, 1.1]): untrained error; init gradient
  alignment of the full-batch local update vs the implicit adjoint; **converged L-BFGS ceiling**
  (`mech_ceiling`); **matched-budget GD baseline** (same epochs, lr = η — the local rule is GD at step η
  in the β → 0 limit); the local rule, β = 0.05, η_k = η_ℓ = 1.0 (η/β = 20), 6000 epochs; trajectory
  diagnostics (best error and epoch, first instability epoch, tail jump rate).

### Pilot observations that shaped the protocol (seed 0, amplify unless noted)
- **Plain GD is far from the ceiling.** Attenuate: L-BFGS ceiling 1.8e-4 vs GD(lr 1, 2000 epochs) 4.9e-2 vs
  local(η 0.5, 2000) 7.3e-2. Amplify: 2.2e-3 vs 1.2e-2 vs 2.7e-2. An offline baseline at a fixed GD budget
  would have *understated* representability by two orders of magnitude — the budget-artifact lesson, on
  the representable side this time.
- **Inversion is representable** (ceiling 4.6e-3) via pre-stress (rest lengths up to 1.66× geometry),
  although the 300-epoch GD pilot sat at 1.04 (worse than doing nothing).
- **The gradient flow walks into buckling.** Local rule, η = 0.5: error 2.66e-2 (2k) → 1.69e-2 (4k) →
  1.06e-2 (6k) → 6.67e-3 (8k) → **0.62 at 10k epochs with instability events appearing** (jump rate 0.03
  over the run, concentrated at the end). η = 2.0 destabilizes within the first epochs (jump rate 0.97);
  η = 1.0 and 1.5 are stable through 1000 epochs. Hence: the tail jump rate (last 20 % of epochs) is the
  multistability gate, best-epoch error is reported beside the final, and GD's failed epochs are counted.

### Headline run (6000 epochs, 3 seeds, 3 tasks)
__TABLE__

### d-map (amplify, seed 0, 1000 epochs): where on the deformation axis does the rule work?
__DMAP__

---

## Interpretation
__INTERP__
