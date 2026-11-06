# Findings — read noise in the elastic two-read rule (commons H5, first mechanical data point)

**Date:** 2026-11-06
**Question.** The mechanical coupled-learning update differences two physical reads of each bond's
extension (free, nudged) and divides by β. Commons hypothesis **H5** says such two-read contrastive rules
are read-noise-fragile through the `η/β` amplification of *independent* read noise, while a common-mode
read error cancels; `07` confirmed it on the flow/resistor rule, `05`'s chemical rule leaned tolerant.
This is the first data point on an **elastic** substrate, run on Experiment B0's amplify task (the rule
learns it cleanly for 2000 epochs; B0 found the late buckling at ~4400 epochs).
**Verdict:** **H5-MECHANISM-CONFIRMED** — independent-read wall at σ = 1e-3 (both β), common-mode: no wall
up to σ = 1e-2 (gap ≥ 10×, bounded by the sweep). **β-scaling unresolved** at decade resolution (both
independent walls fall in the same decade).
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
same seed-and-β σ = 0 error) exceeds 2.

### The signal the noise competes with (init, per-bond |e_nudged − e_free|)

| seed | β | median | max |
|---|---|---|---|
| 0 | 0.05 | 9.2e-05 | 1.5e-03 |
| 0 | 0.20 | 3.7e-04 | 5.9e-03 |
| 1 | 0.05 | 7.4e-05 | 1.1e-03 |
| 1 | 0.20 | 2.9e-04 | 4.5e-03 |

### Error relative to the σ = 0 control (final error / control; `tail` = instability rate in the last 20 % of epochs)

| σ | independent, β 0.05 (s0 / s1) | independent, β 0.2 (s0 / s1) | common, β 0.05 | common, β 0.2 |
|---|---|---|---|---|
| 0 (control) | 1.69e-02 / 6.48e-03 (abs) | 1.65e-02 / 6.32e-03 (abs) | — | — |
| 1e-5 | 0.98 / 0.99 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-4 | 1.14 / 1.60 | 1.02 / 1.13 | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-3 | **59 / 154** (tail 1.00) | **61 / 158** (tail 1.00) | 1.00 / 1.00 | 1.00 / 1.00 |
| 1e-2 | 59 / 154 (tail 1.00) | 61 / 158 (tail 1.00) | 1.00 / 1.00 | 1.00 / 1.00 |

Walls: independent **1e-3** at β = 0.05 and at β = 0.2; common-mode **none** within the sweep. At the
independent wall the *best* error along the trajectory is still good or better than the control
(5.4e-3 / 1.3e-2 at β 0.05; 5.6e-3 / 9.5e-3 at β 0.2) before the tail instability rate reaches 1.00: the
lattice **buckles** and the final error is 1.0 (no solvable readout).

---

## Interpretation

**The two-read mechanism is confirmed on the elastic medium.** Common-mode read error cancels exactly in
the symmetric-spring rest-length update (`Δℓ ∝ k(e_n − e_f)`) and nearly so in the stiffness update, and the
sweep shows precisely that: no degradation at any σ, up to 100× the per-bond signal. Independent noise is
amplified by `1/β` and forecloses at σ ≈ 1e-3 — about **10× the median per-bond contrast** and comparable to
its maximum. That the rule tolerates noise of the order of its own per-bond signal (σ = 1e-4 costs ≤ 1.6×)
is the pattern-averaging over 6 patterns and 33 bonds doing its work.

**The failure mode is new relative to `07` and `02`: noise does not misdirect learning, it buckles the
lattice.** At the wall the trajectory first reaches a *better* best error than the clean run, then the
noise-driven random walk in rest lengths carries the lattice across the stability boundary and every
subsequent relaxation fails. On an elastic substrate, read noise converts into *mechanical* instability —
the same wall B2 and B0 found by other routes. A hardware build would literally buckle, not merely learn
slowly.

**β-scaling is not resolved here.** H5 predicts the independent wall ∝ β; a 4× change in β (0.05 → 0.2)
moves a decade-resolution wall by less than one decade, so both walls land at 1e-3. The verdict's
β-scaling leg is therefore satisfied trivially (`beta_scaling_resolved = False` in the JSON), and the
CONFIRMED status rests on the independent-vs-common gap. A finer σ grid (half-decades) around 1e-3, or
β ∈ {0.02, 0.5}, would resolve it; the degradation at 1e-4 is already smaller at β = 0.2 (1.02 / 1.13) than
at 0.05 (1.14 / 1.60), consistent with the prediction.

**Where this sits in the portfolio.** Third confirmed-fragile data point for the two-read rule (`02`
memristive, `07` flow, now `01` elastic), with `05`'s chemical rule still the exception; gradient-flow
structure (which this substrate has) again does not confer robustness. The σ-multiplier is *not*
commensurable with the other projects' (metric-dependent walls); what transfers is the *side* (fragile),
the mechanism evidence (independent ≫ common-mode), and the new observation that on a mechanical medium the
noise tax is paid as buckling. For the compliant-medium hardware route (`hardware-paths` Route B / H4) this
sharpens the requirement: a single-read or in-fabric-reference rule is needed *and* the read-noise budget
must be set by the buckling margin, not by learning speed.

**Caveats.** Two seeds, one task, one lattice size, decade σ grid, 2000 epochs (before B0's clean-run
buckling onset); "realistic" σ for a printed/compliant lattice is unknown — in these units 1e-3 is a
millistrain-scale readout error on a unit lattice spacing.
