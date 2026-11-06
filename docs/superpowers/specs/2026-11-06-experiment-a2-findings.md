# Findings — Experiment A2: the β-law made visible, and the Poiseuille law run as a headline

**Date:** 2026-11-06
**Scope:** two completeness items Experiment A's findings left open: (1) the β-tradeoff was "barely
visible" on A's uniform 3×3 net; (2) the Hagen–Poiseuille law was implemented but never run.
**Verdicts:** β-law **measured: 1 − ρ ∝ β^2.00** (11 points, no numerical floor to β = 1e-6).
Poiseuille **PASS** on both of A's tasks (allostery, regression). Audit 100 % physics-local.

> **Observed** = measured; **Interpretation** separate. `python experiment_a2.py`, ~1 s.

---

## Observed

### β-sweep on a heterogeneous 5×5 resistor net (random conductances U[0.2, 5], regression task, init)

| β | sign-match | ρ | 1 − ρ |
|---|---|---|---|
| 0.5 | 1.000 | 0.96395 | 3.60e-02 |
| 0.2 | 1.000 | 0.99404 | 5.96e-03 |
| 0.1 | 1.000 | 0.99851 | 1.49e-03 |
| 0.05 | 1.000 | 0.99963 | 3.73e-04 |
| 0.02 | 1.000 | 0.99994 | 5.96e-05 |
| 0.01 | 1.000 | 0.999985 | 1.49e-05 |
| 0.005 | 1.000 | 0.9999963 | 3.72e-06 |
| 0.002 | 1.000 | 0.9999994 | 5.95e-07 |
| 0.001 | 1.000 | 0.99999985 | 1.49e-07 |
| 1e-4 | 1.000 | 1.0000000 | 1.49e-09 |
| 1e-5 | 1.000 | 1.0000000 | 1.49e-11 |
| 1e-6 | 1.000 | 1.0000000 | 1.49e-13 |

Log–log fit: **1 − ρ = c·β^2.00**. The ratio between successive decades is 100.0 to three digits
throughout; no floor appears in double precision down to β = 1e-6.

### Poiseuille law, A's tasks (g = r⁴, trainable radius r ∈ [0.3, 2.0], β = 0.01, η = lr = 0.5, 400 epochs)

| task | local error | baseline error | sign (all edges) | sign (non-zero-gradient edges) | ρ | r range | verdict |
|---|---|---|---|---|---|---|---|
| allostery | 1.20e-29 | 2.20e-31 | 0.83 | **1.00** | 1.000 | [0.92, 1.04] | **PASS** |
| regression | 1.06e-27 | 4.74e-17 | 1.00 | 1.00 | 1.000 | [0.86, 1.11] | **PASS** |

Both tasks are solved to machine precision by both the local rule and the offline baseline.

### Two verdict-apparatus notes (set after a first run; recorded, not hidden)
- The first A2 run returned **FORECLOSE** on allostery because A's `verdict()` is purely relative and
  1.2e-29 > 1.15 × 2.2e-31 — a comparison of two machine zeros. `verdict()` gained an optional
  `abs_floor` (default 0, so Experiment A's rule is byte-identical); A2 sets it to 1e-6 normalized MSE.
- The all-edge sign-match at init was 0.83 because 2 of the 12 edges of the symmetric 3×3 allostery
  lattice (the middle-row edges) have an **exactly zero** gradient (|∂L/∂g| ≈ 1e-18 of the maximum) while
  the local update there is the O(β) second-order term; the sign of a zero is noise.
  `gradient_alignment` now also reports `sign_match_significant` (components with |grad| > 1e-9·max);
  A2's verdict uses it. Both metrics are printed.

---

## Interpretation

**The β → 0 limit is quadratic, and it is clean.** For a linear network the nudged drop is
`ΔV_n = ΔV_f + β δ` with δ the linear response, so the per-edge contrast is
`(ΔV_f² − ΔV_n²)/β = −2 ΔV_f δ − β δ²`: the first term *is* the gradient and the deviation is first order
in β. ρ measures an angle, and an angle squared is second order — hence 1 − ρ ∝ β², now measured to three
digits over six decades. The "smaller β → more numerical noise" tradeoff that A's findings anticipated is
**not present** on a clean linear substrate within double precision; it only appears with read noise in
the relaxations (the H5 question, which `07` has answered for this rule: fragile). A's statement should be
read as: *on noiseless hardware-free simulation β can be made arbitrarily small; on hardware β trades
alignment against read-noise amplification `η/β`.*

**The Poiseuille (flow) form of the rule is a PASS**, with the `∂g/∂r = 4Cr³` chain rule carried through
the local update (the `edge_energy_grad` trap the commons promotion closed). This is the form `07` prints;
the result is unsurprising but closes an open item and gives the hardware arm a reproducible software
reference point (radii move only ~±10 % to solve both tasks).

**Caveat.** Both Poiseuille tasks are solved to machine zero, so the *comparison* to the baseline is not
informative beyond "both solve it"; the PASS is carried by the alignment (ρ = 1.000) and by the error
being at the floor. The `abs_floor` and significant-sign metric were introduced after seeing the first
run; they are measurement fixes, not threshold moves, and A's own verdict is unchanged.
