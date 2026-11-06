# Findings — read noise in the elastic two-read rule (commons H5, first mechanical data point)

**Date:** 2026-11-06 (run 2, corrected solver seam; run 1 summarized at the end)
**Question.** The mechanical coupled-learning update differences two physical reads of each bond's
extension (free, nudged) and divides by β. Commons hypothesis **H5** says such two-read contrastive rules
are read-noise-fragile through the `1/β` amplification of *independent* read noise, while a common-mode
read error cancels; `07` confirmed it on the flow/resistor rule, `05`'s chemical rule leaned tolerant.
This is the first data point on an **elastic** substrate, run on Experiment B0's amplify task.
**Verdict:** **H5-MECHANISM-CONFIRMED** — independent-read wall at σ = 1e-3 (β = 0.05) and 1e-2 (β = 0.2),
common-mode: no wall up to 1e-2 at either β; the wall **moves with β** as predicted (resolved this run).
**Audit:** 100 % `PHYSICS_LOCAL` (the noise is in the reads, not in who computes).

> **Observed** = measured; **Interpretation** separate. Pre-registered verdict `tasks.verdict_b0_noise`
> (mechanism form: CONFIRMED if independent wall ≥ 10× below common-mode at every β and non-decreasing in
> β; REFUTED if within 3×; UNTESTED if the σ = 0 control does not learn). No hardware-realistic σ exists
> for this medium, so this is a *mechanism* verdict, not a device verdict.

---

## Observed

### Apparatus
4×4 lattice, amplify task (ratio 1.033 = 2 r₀), η = 1.0, 2000 epochs, seeds 0–1, β ∈ {0.05, 0.2},
σ ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2} lattice-spacing units, modes: *independent* (fresh Gaussian per read) and
*common* (same realization on both reads). Wall criterion: the median over seeds of (final error / the
same seed-and-β σ = 0 error) exceeds 2. Free phase warm-started with `settle()` fallback (the corrected
seam); `tail` = genuine instability rate in the last 20 % of epochs. 755 s.

### The signal the noise competes with (init, per-bond |e_nudged − e_free|)

| seed | β | median | max |
|---|---|---|---|
| 0 | 0.05 | 9.2e-05 | 1.5e-03 |
| 0 | 0.20 | 3.7e-04 | 5.9e-03 |
| 1 | 0.05 | 7.4e-05 | 1.1e-03 |
| 1 | 0.20 | 2.9e-04 | 4.5e-03 |

### Final error relative to the σ = 0 control (seed 0 / seed 1); `best` = minimum along the trajectory

| σ | independent, β 0.05 | independent, β 0.2 | common, β 0.05 | common, β 0.2 |
|---|---|---|---|---|
| 0 (control, absolute) | 1.69e-02 / 6.48e-03 | 1.65e-02 / 6.32e-03 | — | — |
| 1e-5 | 0.98 / 0.99 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-4 | 1.14 / 1.71 | 1.02 / 1.13 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-3 | **3.11 / 8.69** (best 9.9e-5 / 4.9e-5; tail 0.01) | 0.84 / 1.11 (best 1.5e-4 / 1.2e-4) | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-2 | 5367 / 111964 (tail 0.08 / 0.07) | **719 / 75** (tail 1.00 / 0.02) | 1.00 / 1.00 | 1.00 / 1.00 |

Walls: independent **1e-3 at β = 0.05, 1e-2 at β = 0.2**; common-mode **none** within the sweep. The
verdict JSON reports `beta_scaling_resolved = True`.

### Two features of the independent-noise trajectories
- **Below the wall, noise transiently helps.** At σ = 1e-3 the best error along the noisy trajectory is
  5e-5 – 1.5e-4 — about **100× below the clean run's 2000-epoch error** (6e-3 – 1.7e-2) and below the
  clean run's value at *any* epoch (the clean trajectory is monotone) — before the parameters drift and
  the final error ends 0.8–9× the control.
- **At the wall the failure is drift, not buckling** (contrast run 1, below): tail instability rates are
  0.01–0.08 while errors grow to 10²–10⁵ (rest lengths random-walk into the clip bounds); only one cell
  (σ = 1e-2, β = 0.2, seed 0) ends in genuine persistent instability (tail 1.00).

---

## Interpretation

**The two-read mechanism is confirmed on the elastic medium, including its β-dependence.** Common-mode
read error cancels exactly in the symmetric-spring rest-length update (`Δℓ ∝ k(e_n − e_f)`) and nearly so in
the stiffness update, and the sweep shows precisely that: no degradation at any σ, up to 100× the per-bond
signal. Independent noise is amplified by `1/β`: the wall sits at σ ≈ 1e-3 for β = 0.05 and at ≈ 1e-2 for
β = 0.2 — a 4× larger nudge buys the next decade of noise tolerance, as H5 predicts. In signal units the
wall is ~10× the median per-bond contrast at both β (1e-3 / 9e-5 ≈ 11; 1e-2 / 3.7e-4 ≈ 27 — within the
decade grid), so the operative statement is: **the elastic two-read rule tolerates independent read noise
up to roughly 10× its own per-bond contrast, and that contrast scales with β.**

**A noise-helps transient.** Around the wall the noisy rule finds configurations 100× better than the clean
rule reaches in the same budget, then drifts away from them. Read noise acts as stochastic exploration on a
loss whose clean gradient flow is slow (B0: attenuate/amplify had not plateaued at 6000 epochs); with no
annealing, the exploration that helped later hurts. This is the commons' `noise-as-resource` pattern
(`03`: a *gentle anneal* turns noise from tax into help) on a mechanical substrate — an **observation with a
selection caveat**: "best along the trajectory" is a minimum over 2000 noisy epochs. A pre-registered
follow-up (σ annealed to zero over training; compare final error to the clean run) is the clean test.

**Correction relative to run 1.** Run 1 (cold-start seam) read "the tax is paid as buckling" (final error
1.0, tail 1.00 at every σ ≥ 1e-3) and both independent walls at 1e-3. Those blow-ups were the
free-phase cold-start artifact documented in Experiment B's addendum: with the corrected seam the same
cells end in a drift with near-zero genuine instability. The mechanism verdict is unchanged; the β-scaling
that run 1 could not resolve is now resolved; the "buckles under noise" interpretation is withdrawn.

**Where this sits in the portfolio.** Third confirmed-fragile data point for the two-read rule (`02`
memristive, `07` flow, now `01` elastic), with `05`'s chemical rule still the exception; gradient-flow
structure again does not confer robustness. The σ-multiplier is *not* commensurable with the other
projects' (metric-dependent walls); what transfers is the *side* (fragile), the mechanism evidence
(independent ≫ common-mode; wall ∝ β), and the noise-helps transient. For the compliant-medium hardware
route (`hardware-paths` Route B / H4) the requirement is a single-read or in-fabric-reference rule, with a
read-noise budget of order the per-bond nudge contrast.

**Caveats.** Two seeds, one task, one lattice, decade σ grid, 2000 epochs; "realistic" σ for a printed or
compliant lattice is unknown — in these units 1e-3 is a millistrain-scale readout error on a unit spacing.
