# Findings — Experiment B2: is B's representability FORECLOSE a budget / objective artifact?

**Date:** 2026-11-05
**Question.** Experiment B (2026-06-19) declared `FORECLOSE-REPRESENTABILITY` for parity-3 on the elastic
lattice from an offline baseline that minimized MSE to **fixed-scale** targets (±0.5·d) by **plain
gradient descent** for a **fixed 400-epoch budget**. The portfolio commons has since recorded that each of
these can manufacture a false foreclosure (`lessons-ledger#foreclosure-can-be-a-budget-artifact`; `03`'s
parity-4 "foreclosure" dissolved at 3000 steps). B2 asks: does a scale-free objective, a better-converged
optimizer with a saturation record, and a second optimizer class change B's answer?
**Verdict:** __VERDICT__
**Audit:** every update in B2 is `OFFLINE_GRADIENT` by construction (`all_offline = True`) — this is a
representability probe, not a learning run; nothing here bears on the audit question.

> Method discipline: **Observed** = measured numbers; **Interpretation** kept separate. Pre-registered
> verdict logic: `tasks.verdict_b2` (REPRESENTABLE / FORECLOSE-REPRESENTABILITY-CONFIRMED / AMBIGUOUS;
> thresholds SNR ≥ 3, unstable-seed fraction ≤ 0.2, saturation gap ≤ 0.02).

---

## Observed

### Apparatus
- **Objective (scale-free).** Any readout over the 8 patterns is exactly a Walsh expansion
  `r = a₀ + Σ aᵢsᵢ + Σ aᵢⱼsᵢsⱼ + a₁₂₃ s₁s₂s₃`; parity lives entirely in `a₁₂₃`. **Parity purity**
  `a₁₂₃² / Σⱼ≥₁ aⱼ²` ∈ [0,1] is 1 iff the readout *is* parity up to offset and scale; the **parity cosine**
  `a₁₂₃ / √Σⱼ≥₁ aⱼ²` is its signed square root. Both are differentiated through the implicit-function
  adjoint (`mech_readout_adjoint`, FD-checked at the level of the full θ→loss map in
  `test_experiment_b2.py`).
- **Optimizers.** L-BFGS-B (scipy) on the cosine and, as a control, on B's original MSE objective, from
  B's random inits (k ∈ U[0.5,2], ℓ ∈ ℓ₀·U[0.9,1.1]), bounds = B's clip bounds (k ∈ [1e-2,1e2],
  ℓ ∈ [0.1,5]); budget 800 iterations with purity recorded at 50 / 200 / 800; failed equilibrium solves
  return a quadratic barrier back to the last solvable point; restarts from the best point on line-search
  abort. Differential evolution (gradient-free; popsize 10, 60 generations; k ∈ [0.2,5], ℓ ∈ ℓ₀·[0.8,1.2]).
- **Sweep.** 4×4 lattice, d ∈ {0.05, 0.10, 0.15, 0.20}, seeds 0–4 (L-BFGS, both objectives), DE at
  d ∈ {0.10, 0.15}, seed 0.
- **Diagnostics.** Two-sided separation at the optimum (the verdict quantity); unstable-seed fraction
  (lattices unsolvable even at init); eval-fail rate; **wall-pinned** fraction (cells whose last 20
  evaluations include failed solves — the optimum sits on the buckling boundary).

### The untrained lattice's parity channel scales as d³ (uniform k = 1, unstressed ℓ = ℓ₀)

| d | linear `max|aᵢ|` | quadratic `max|aᵢⱼ|` | parity `|a₁₂₃|` | parity / linear | purity |
|---|---|---|---|---|---|
| 0.02 | 1.261e-02 | 9.609e-05 | 3.647e-06 | 2.9e-04 | 0.0000 |
| 0.05 | 3.158e-02 | 6.066e-04 | 5.738e-05 | 1.8e-03 | 0.0000 |
| 0.10 | 6.346e-02 | 2.515e-03 | 4.706e-04 | 7.4e-03 | 0.0000 |
| 0.15 | 9.599e-02 | 6.020e-03 | 1.661e-03 | 1.7e-02 | 0.0002 |
| 0.20 | 1.296e-01 | 1.173e-02 | 4.227e-03 | 3.3e-02 | 0.0006 |
| 0.30 | — | — | — | — | **buckled** (indefinite tangent stiffness) |

Log–log fits over the stable points: **linear ∝ d^1.01, parity ∝ d^3.06** (4×4); d^1.00 / d^3.03 (6×6).
Extrapolated parity = linear crossover: **d ≈ 1.08** lattice spacings (4×4), **d ≈ 2.36** (6×6); the
lattice buckles at d ≈ 0.25–0.30.

### Run 1 — as first pre-registered (purity objective, flat failure penalty): AMBIGUOUS, apparatus-limited
Kept for the record (`scratchpad` log `b2_run1.log`). Every cell terminated early: the purity gradient is
∝ a₁₂₃ and vanishes on the parity-free untrained lattice (4 cells "converged" at iteration 0 by the
projected-gradient test), and the constant penalty on failed solves aborted L-BFGS's line search
(ABNORMAL) in 13 of 24 cells after 1–34 iterations. Best purity found 0.134 (L-BFGS) / 0.404 (DE, 85 % of
its evaluations unsolvable); **every separation negative**. Verdict AMBIGUOUS solely because saturation
could not be measured. These are measurement-apparatus defects, fixed before run 2 (cosine objective,
barrier + restarts, seed-based stability gate); the verdict thresholds were not changed.

### Run 2 — revised apparatus
__RUN2__

---

## Interpretation
__INTERP__
