# Findings — Experiment B3: parity-3 vs deformation, with multistability measured

**Date:** 2026-11-06
**Question.** Experiment B's June headline attributed its foreclosure partly to multistability inferred from
solver crashes; those crashes were a cold-start artifact (B's 2026-11-06 addendum), and under the corrected
seam B at d = 0.1 is a pure representability foreclosure. B2 shows the parity channel of the lattice growing
large only at d ≳ 0.3. So the question B meant to ask is still open: **at deformations where the parity
channel is large, does the local two-phase rule separate parity-3 — and if so, is that separation a property
of the lattice or of which basin the trajectory happens to sit in?** B3 trains at d ∈ {0.1, …, 0.5} and then
**validates** each learned configuration by multi-start minimization: distinct energy minima per pattern,
the readout at the lowest-energy minimum, and the smallest tangent-stiffness eigenvalue there.
**Verdicts:** __VERDICTS__
**Audit:** headline training 100 % `PHYSICS_LOCAL`; validation is a measurement.

> **Observed** = measured; **Interpretation** separate. Pre-registered `tasks.verdict_b3` per d:
> PASS-ROBUST (lowest-energy separation > 0, SNR ≥ 3, stable, every seed) · MULTISTABLE-HYSTERETIC (≥ 50 %
> of patterns multistable on a majority of seeds, and some seed's trajectory separation > 0 while its
> lowest-energy separation ≤ 0) · FORECLOSE-REPRESENTABILITY (unique minima, negative separation) ·
> AMBIGUOUS.

---

## Observed

### Pilot that motivated B3 (Experiment B driver, corrected seam, d-sweep; 3 seeds × 400 epochs)

| d | local_sep (worst / per seed) | baseline_sep (GD) | measured min eigenvalue (per seed) | rescues |
|---|---|---|---|---|
| 0.1 | −7.3e-02 (−0.072 / −0.062 / −0.073) | −0.073 / −0.059 / −0.044 | +0.029 / +0.023 / +0.034 | 0 |
| 0.2 | −8.4e-02 (−0.057 / −0.084 / −0.080) | −0.047 / −0.060 / −0.073 | +0.010 / +0.008 / +0.027 | 0–1 |
| 0.3 | −1.0e-02 (**+0.056** / **+0.008** / −0.010) | −0.240 / −0.106 / −0.011 | +0.0008 / +0.0006 / +0.0001 | 2–6 |
| 0.4 | −5.6e-04 (**+0.331** / −0.00003 / −0.0006) | −inf / −inf / −0.042 | +0.0006 / +0.0001 / **−0.148** | 5–10 |
| 0.5 | **+7.2e-02** (**+0.370** / **+0.452** / **+0.072**) | −inf / −inf / −inf | **−0.275** / +0.0007 / **−1.238** | 5–15 |

At d = 0.5 the trained run's own readouts separate parity-3 on **all three seeds**, while the offline GD
baseline cannot be evaluated (its trajectory reaches genuinely unsolvable states). The stability margin
collapses from 0.03 at d = 0.1 to ~1e-3–1e-4 at d ≥ 0.3, and negative eigenvalues at d ≥ 0.4 show Newton
returning **saddles** along the warm-started path (fixed in the seam before the B3 run: the free phase now
rejects non-minima).

**Multi-start validation of the d = 0.5 configurations** (settle() from the reference geometry and six
perturbed starts, per pattern):

| seed | trajectory separation | distinct minima per pattern | lowest-energy separation |
|---|---|---|---|
| 0 | +0.370 | 3, 3, 5, 4, 2, 2, 1, 2 | **−1.60** |
| 1 | +0.452 | 1, 2, 3, 3, 1, 1, 1, 2 | **−0.75** |
| 2 | +0.072 | 2, 3, 4, 3, 2, 1, 2, 2 | **−0.41** |

Every lowest-energy minimum is a genuine minimum (eigenvalues +0.003 … +0.03), and several have readouts of
1–4 lattice spacings (the lattice is folded).

### Headline run (`python experiment_b3.py`, corrected seam incl. saddle rejection)
__TABLE__

---

## Interpretation
__INTERP__
