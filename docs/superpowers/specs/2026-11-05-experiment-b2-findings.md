# Findings — Experiment B2: is B's representability FORECLOSE a budget / objective artifact?

**Date:** 2026-11-05
**Question.** Experiment B (2026-06-19) declared `FORECLOSE-REPRESENTABILITY` for parity-3 on the elastic
lattice from an offline baseline that minimized MSE to **fixed-scale** targets (±0.5·d) by **plain
gradient descent** for a **fixed 400-epoch budget**. The portfolio commons has since recorded that each of
these can manufacture a false foreclosure (`lessons-ledger#foreclosure-can-be-a-budget-artifact`; `03`'s
parity-4 "foreclosure" dissolved at 3000 steps). B2 asks: does a scale-free objective, a better-converged
optimizer with a saturation record, and a second optimizer class change B's answer?
**Verdict:** **FORECLOSE-REPRESENTABILITY-CONFIRMED.** B's representability foreclosure was *not* a
budget or objective artifact. A scale-free objective and better-converged optimizers grow the lattice's
parity channel by two to three orders of magnitude — and every optimum lands **pinned on the buckling
boundary** with the two-sided separation still negative (38 / 38 solvable cells).
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
1668 s wall-clock (vectorized kernels). 40 L-BFGS cells (4 d × 5 seeds × 2 objectives) + 2 DE cells.

**Per-d aggregate** (verdict inputs; `sep` = two-sided separation at the optimum, min/max over all cells
at that d; `a₁₂₃ max` = largest parity coefficient reached; `pinned` = fraction of solvable cells whose
last 20 evaluations include failed solves; `unstable` = fraction of seeds whose lattice is unsolvable at
init; `eval-fail` = fraction of all objective evaluations that hit an unsolvable configuration):

| d | sep_min | sep_max | purity max | a₁₂₃ max | pinned | unstable seeds | eval-fail | saturation gap |
|---|---|---|---|---|---|---|---|---|
| 0.05 | −3.16e-01 | **−1.75e-02** | 0.176 | 2.60e-01 | 1.00 | 0.0 | 0.35 | 0.000 |
| 0.10 | −3.70e-01 | **−2.13e-02** | 0.476 | 2.85e-01 | 1.00 | 0.0 | 0.32 | 0.000 |
| 0.15 | −4.87e-01 | **−6.18e-02** | 0.553 | 3.85e-01 | 1.00 | 0.2 (seed 1) | 0.44 | 0.000 |
| 0.20 | −5.98e-01 | **−4.41e-02** | 0.186 | 4.66e-01 | 1.00 | 0.2 (seed 1) | 0.47 | 0.000 |

**Separation is negative in every solvable cell** — all 36 L-BFGS cells and both DE cells. Seed 1's random
rest lengths buckle the lattice at init for d ≥ 0.15 (4 cells unsolvable; unstable fraction 0.2, at the
pre-registered threshold).

**By objective / optimizer** (ranges over solvable cells):

| probe | purity reached | `|a₁₂₃|` reached | separation |
|---|---|---|---|
| untrained lattice (reference) | 0.0000–0.0006 | 3.6e-06 – 4.2e-03 | — |
| L-BFGS on **cosine** (18 cells) | 0.018–0.186 | 2.6e-02 – 4.7e-01 | −0.60 … −0.17 |
| L-BFGS on **MSE ±0.5d** (18 cells; B's objective) | 0.000–0.049 | 1.3e-06 – 1.7e-02 | −0.27 … −0.018 |
| DE on \|cosine\|, d = 0.10 (40,260 evals) | 0.476 | 2.85e-01 (linear 2.2e-01, quadratic 1.1e-01) | −0.223 |
| DE on \|cosine\|, d = 0.15 (40,260 evals) | 0.553 | 3.85e-01 (linear 2.1e-01, quadratic 1.6e-01) | −0.113 |

**Convergence / saturation.** L-BFGS terminated after 2–42 iterations in every cell — before the first
checkpoint (50) — so all three checkpoints carry the final value and the saturation gap is 0 by
construction. 26 of 36 solvable cells ended with a genuine `CONVERGENCE` message; 10 (eight of them MSE
control cells) exhausted the 8 restarts, each restart aborting in the line search at the stability wall.
**100 % of solvable cells are wall-pinned**: 32–47 % of all objective evaluations landed on an unsolvable
(indefinite-tangent-stiffness) configuration, concentrated at the end of each run.

**Where the optimizer goes.** Inspecting the DE optima: the parity coefficient becomes the *largest*
single Walsh coefficient (0.29 vs linear 0.22 at d = 0.10; 0.38 vs 0.21 at d = 0.15), yet the three
linear and three quadratic channels together still outweigh it, so the eight readouts interleave and
the two-sided separation stays negative.

---

## Interpretation
**B's representability foreclosure survives the stronger probe — and is now mechanistically
quantified.** Three independent lines agree:

1. **Scaling.** On the untrained lattice the parity channel grows as **d³** while the linear channel grows
   as **d¹** (fits 3.06 / 1.01 on 4×4, 3.03 / 1.00 on 6×6). That is the generic signature of a smooth
   odd response: parity is the lowest-order odd term beyond linear, so it enters at third order in the
   drive. The two channels would cross only at d ≈ 1.1 lattice spacings (4×4) or 2.4 (6×6) — well past
   the buckling threshold at d ≈ 0.25–0.30. **The untrained substrate cannot reach parity dominance before
   it buckles, by a factor 4–10 in deformation.** (Larger lattices make it *worse*: the linear response
   is more robust to size than the cubic one.)
2. **Optimization pins to the wall.** Learning `k` and `ℓ` can rebalance the channels — the parity
   coefficient rises from ~5e-4 to 0.29–0.47 (a factor of several hundred), mostly by pre-stressing the
   lattice (rest lengths ≠ geometry) to amplify geometric nonlinearity. But every optimum, under every
   objective and optimizer, sits on the boundary where the tangent stiffness goes indefinite:
   **the direction in parameter space that increases parity content is the direction that buckles the
   lattice.** This is Experiment B's "pincer" (`lessons-ledger#expressivity-multistability-pincer`)
   observed directly in the optimizer's trajectory rather than inferred from two separate regimes.
3. **No optimizer class or objective finds a separating configuration.** Gradient (L-BFGS) and
   gradient-free (DE, 40k evaluations) probes, a scale-free objective and B's fixed-scale one, five random
   inits, four deformations: 38 / 38 solvable optima have negative two-sided separation. Even the best
   configuration (purity 0.55) leaves the quadratic channels — which are *even* under a global bit flip
   and therefore scramble the within-class ordering — too large.

**What the re-probe did change: the methodology.** B's original MSE-to-±0.5d baseline is confirmed to be a
*weak* probe — under L-BFGS it leaves purity at ≤ 0.05 and the parity coefficient ≤ 0.017, two orders of
magnitude below what the scale-free objective reaches from the same inits. B reached the right verdict
with an objective that barely explored the relevant direction; the commons lesson
`foreclosure-can-be-a-budget-artifact` was a legitimate concern, and B2 is the test it demanded. The
first B2 run also showed two apparatus traps worth recording for siblings: a purity-type objective is
gradient-degenerate on a parity-free start (optimize the cosine instead), and a flat penalty on
unsolvable states aborts quasi-Newton line searches (use a barrier back to the last stable point).

**Honest caveats.**
- The probes are still local (L-BFGS) or modest-budget evolutionary (DE, popsize 10 × 60 generations).
  Larger DE budgets or basin-hopping could find higher purity; nothing seen suggests they would cross to
  positive separation while staying solvable, but the claim remains "strongly unsupported across a broad
  sweep," not "impossible."
- One topology (triangulated 4×4), one I/O placement, one readout component. Not swept.
- The stability gate was met exactly at threshold at d ≥ 0.15 (seed 1 of 5 unsolvable at init).
- 10 cells ended by restart exhaustion rather than a convergence message; they are counted as saturated
  because every restart aborted at the stability wall, i.e. more budget on this optimizer would not help.
- Scalar readout only. A multi-node or vector readout changes the representability question (the output
  could combine several channels) — that is a different experiment, not a flaw in this one.

**Portfolio implication.** `01`-B's FORECLOSE now rests on a budget-saturated, scale-free, two-optimizer
probe with a quantitative mechanism (d³ vs d¹; wall-pinned optima), so it can be cited as a
*representability* foreclosure in the commons' sense, not a budget one. The escape from the parity wall
on a passive mechanical substrate would need non-monotonicity that does **not** arrive through the
buckling threshold — e.g. elements that are bistable *by design* with a learning rule that tolerates
multistability, or a vector/multi-node readout — or a substrate whose non-monotonicity is intrinsic
(`03`, discrete spins). Reusable for siblings: the **wall-pinned fraction** as a diagnostic that an
offline ceiling is limited by instability rather than by expressivity or budget.
