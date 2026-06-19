# Findings — Experiment A: coupled learning in a passive resistor network

**Date:** 2026-06-18
**Verdict:** **PASS** (both tasks). The local, physics-driven rule drives task error to (and below) the
offline optimizer baseline, and its per-step update is a near-perfect image of the true loss gradient.
**Audit:** 100% physics-local for both headline runs — *the physics computed the update.*

> Method discipline: the **Observed** section is strictly measured numbers; the **Interpretation**
> section is kept separate. See `2026-06-18-coupled-learning-networks-design.md` for the pre-registered
> verdicts this is judged against.

---

## Observed

### Test suite
24 tests passing (`python -m pytest -q`), one module per source module. The keystone
(`test_local_update_aligns_with_true_gradient_as_beta_shrinks`) and the adjoint cross-check
(`test_adjoint_matches_finite_difference`, abs=1e-5) both pass.

### Headline verdict table (`python experiment_a.py`)

| task | local_error | baseline_error | sign_match | ρ | verdict | physics-local |
|------|-------------|----------------|-----------|-----|---------|---------------|
| allostery  | 3.6736e-13 | 4.4890e-08 | 1.00 | 1.000 | **PASS** | 1.0 |
| regression | 1.2610e-04 | 9.2315e-04 | 1.00 | 1.000 | **PASS** | 1.0 |

- `local_error` = normalized MSE of the network trained by the local coupled-learning rule.
- `baseline_error` = normalized MSE of the offline adjoint-gradient-descent optimizer ("CPU computes
  the update").
- On both tasks the local rule's final error is **at or below** the offline baseline.
- `audit_fraction_physics_local == 1.0` for both headline runs; top-level `run()["audit"]
  ["fraction_physics_local"] == 1.0`. The baseline used a separate audit log (offline-gradient).

### Gradient alignment vs nudge β (3×3 grid, regression, seed 3)

| β | ρ (local update vs −true gradient) |
|------|------------------------------------|
| 0.2   | 0.99957 |
| 0.05  | 0.99997 |
| 0.01  | 0.999999 |
| 0.002 | 1.0000000 |

At β = 0.002: sign_match = 1.0, ρ = 1.0. Alignment increases monotonically as β shrinks. The adjoint
gradient was independently validated against a central finite-difference gradient (no shared code path
with the coupled-learning update).

### Configuration
Resistor (linear, `g = w`) substrate; 3×3 grid networks; coupled-learning nudge β = 0.01, η = 0.5,
400 epochs; baseline GD 400 epochs, lr = 0.5; conductances clipped to [1e-3, 10.0] (passivity enforced
every step). Stack: numpy + pytest (scipy declared but unused in Experiment A).

---

## Interpretation

**The dual-role learning thesis is real in a physical substrate.** A passive resistor network, trained
only by a two-phase rule that uses quantities physically local to each edge (the voltage drop across that
edge in a free and a nudged relaxation), reaches the offline optimizer's task error on both allostery and
regression. The audit log shows every weight change in the headline runs was physics-local — **no CPU
optimizer touched a weight.** This is the cleanest, cheapest affirmative answer to the portfolio's audit
question, *"who computes the update?"*: **the physics did.**

**The mechanism, not just the outcome, is confirmed.** The per-step coupled-learning update aligns with
the exact loss gradient (sign-match 1.0, ρ 1.0 as β→0). This is the equilibrium-propagation prediction
made concrete on this substrate: the local rule is not merely *a* descent direction that happens to work
— it *is* the gradient in the small-nudge limit. Because the adjoint gradient was cross-checked against
finite differences and computed on an independent code path, the alignment is not circular.

**Honest scoping of the result.**
- A's PASS was *near-foregone* — Dillavou et al. (2022) already demonstrated decentralized physics-driven
  learning in resistor hardware. A's value here is (a) validating the shared harness and the audit hook
  that later projects depend on, (b) a clean self-contained reproduction, and (c) an explicit
  gradient-alignment measurement. All three are delivered.
- The substrate is **linear and the tasks are friendly** (representable by a positive-conductance
  network). The local rule reaching the baseline is *learnability*; the baseline reaching low error is
  *representability* — the harness keeps these distinct.
- **A is deliberately not for parity.** Positive conductances obey the maximum principle (the output is
  monotone in the inputs), so parity-3 is foreclosed by passivity — by construction, not by failure. That
  wall is what Experiment B (signed/non-monotone elastic networks) is designed to test.
- The β-sweep tradeoff ("smaller β → better alignment, more numerical noise") was **barely visible** on
  this benign linear net: ρ was already 0.9996 at β = 0.2 and the improvement to 1.0 spans only ~4e-4.
  The alignment claim is robust (the load-bearing assertion is the small-β floor), but the β-dependence
  is not dramatically exhibited here. A more heterogeneous network would show it more clearly.

**Implication for the portfolio.** The harness (solver / coupled-learning rule / audit / verdict) is
validated and extraction-ready. The thesis holds where passivity allows representation; the open question
— whether co-located learning can clear the parity wall via signed, non-monotone state — moves to
**Experiment B**, whose plan must additionally address the multistability risk (free and nudged phases
landing in different basins) and a genuine vector nonlinear-spring solver. For B's keystone, assert a
concrete alignment margin (not just an endpoint inequality), since the β-limit is genuinely fragile under
multistability.
