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
**Verdicts (per d):** **d = 0.1, 0.2: FORECLOSE-REPRESENTABILITY · d = 0.3, 0.4, 0.5: MULTISTABLE-HYSTERETIC.**
Where the parity channel is large enough for the local rule to separate parity-3 (all three seeds at
d = 0.5: +0.29, +0.45, +0.12), the lattice is multistable (50–88 % of patterns have 2–5 minima) and the
separation exists only in the basins the warm-started trajectory selected; at each pattern's lowest-energy
minimum the separation is strongly negative (−0.44 … −0.64). No deformation yields PASS-ROBUST. **Experiment
B's "pincer" is re-established in measured form:** representability foreclosure at small d, basin-selection
(hysteresis) rather than a lattice property at large d.
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
85 s (training 400 epochs, β = 0.01, η = 1; validation: settle() from x₀ + 6 perturbed starts per pattern).
`sep_traj` = two-sided separation of the trained run's own readouts; `sep_lowest` = re-evaluated at each
pattern's lowest-energy minimum; `multi` = fraction of the 8 patterns with ≥ 2 distinct minima; `minEig` =
smallest eigenvalue over the validated minima; `|r|max` = largest lowest-energy readout (lattice units);
`rescues` = Newton points rejected (line-search failure or saddle) and replaced by settle() during training.

| d | seed | sep_traj | sep_lowest | multi | minima per pattern | minEig | \|r\|max | rescues |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | −0.072 | −0.072 | 0.00 | 1 1 1 1 1 1 1 1 | +0.029 | 0.05 | 0 |
| 0.1 | 1 | −0.062 | −0.062 | 0.00 | 1 1 1 1 1 1 1 1 | +0.033 | 0.07 | 0 |
| 0.1 | 2 | −0.073 | −0.073 | 0.00 | 1 1 1 1 1 1 1 1 | +0.035 | 0.06 | 0 |
| 0.2 | 0 | −0.057 | −0.048 | 0.38 | 1 1 2 1 2 1 1 2 | +0.010 | 0.05 | 0 |
| 0.2 | 1 | −0.084 | −0.084 | 0.25 | 1 1 2 2 1 1 1 1 | +0.025 | 0.10 | 1 |
| 0.2 | 2 | −0.080 | −0.093 | 0.25 | 1 1 2 2 1 1 1 1 | +0.022 | 0.12 | 0 |
| 0.3 | 0 | **+0.056** | −0.246 | 0.88 | 2 5 2 2 2 3 2 1 | +0.006 | 1.46 | 6 |
| 0.3 | 1 | **+0.008** | −0.096 | 0.62 | 3 3 3 3 1 1 1 2 | +0.005 | 0.18 | 2 |
| 0.3 | 2 | −0.010 | −0.206 | 0.62 | 2 2 2 3 1 1 2 1 | +0.003 | 0.43 | 5 |
| 0.4 | 0 | **+0.331** | −0.300 | 0.62 | 3 3 5 4 1 1 2 1 | +0.004 | 0.32 | 10 |
| 0.4 | 1 | −0.000 | −0.421 | 0.50 | 2 3 3 2 1 1 1 1 | +0.003 | 1.22 | 8 |
| 0.4 | 2 | **+0.165** | −0.361 | 0.62 | 3 4 3 4 1 1 1 2 | +0.001 | 0.26 | 13 |
| 0.5 | 0 | **+0.293** | −0.441 | 0.62 | 4 3 2 3 2 1 1 1 | +0.004 | 2.09 | 10 |
| 0.5 | 1 | **+0.452** | −0.593 | 0.62 | 2 2 3 2 1 1 1 4 | +0.003 | 0.91 | 9 |
| 0.5 | 2 | **+0.115** | −0.644 | 0.50 | 3 2 1 1 1 2 3 1 | +0.001 | 0.93 | 43 |

Per-d verdicts: 0.1 FORECLOSE-REPRESENTABILITY · 0.2 FORECLOSE-REPRESENTABILITY · 0.3 MULTISTABLE-HYSTERETIC ·
0.4 MULTISTABLE-HYSTERETIC · 0.5 MULTISTABLE-HYSTERETIC. Audit 1.00. Init alignment ρ = 1.000 at every d
(from the Experiment B driver's d-sweep). Multistability is already present at d = 0.2 (25–38 % of patterns
bistable) but below the 50 % gate and without any positive separation.

---

## Interpretation
**The elastic lattice does not separate parity-3 as a property of its configuration at any deformation
probed; what the local rule finds at large d is basin selection.** The ladder is clean and monotone in d:
unique stable minima and negative separation at d = 0.1; bistability appearing in a quarter to a third of
patterns at d = 0.2 with separation still negative in both measures; from d = 0.3 on, most patterns have
several minima, the trained trajectory's own readouts separate on more and more seeds (all three at
d = 0.5), and the separation re-evaluated at the thermodynamically selected (lowest-energy) minima is
*more* negative the larger d. The "learned" separation lives in the history: each pattern's free equilibrium
is whichever well the warm-started path reached, and the two-phase rule, which always compares a nudged state
*in the same well*, happily shapes those wells. Reset the lattice (settle from the reference geometry) and the
separation is gone.

**Experiment B's pincer, re-established honestly.** B's June headline said: the stable regime cannot
represent parity, and the expressive regime is multistable so the two-phase rule is ill-defined. The second
half rested on solver crashes that were artifacts; B3 now establishes it on measured minima: **the parity
channel becomes large exactly where the energy landscape becomes multi-welled** (stability margin falling
from 0.03 to ~1e-3, 2–5 minima per pattern), and in that regime the rule's readout is a path-dependent
variable, not a function of the input. Both halves of the pincer are now data; the mechanism statement
("rule ill-defined") is sharpened to "the readout is hysteretic".

**Is hysteretic separation worth anything?** Physically, a lattice whose state depends on the order of
inputs is a device with memory, and a *protocol* (always approach pattern p from the same previous state)
could make the d = 0.5 separation reproducible. That is not the question B asked (a feasible, robust
separation of the eight inputs by the lattice), and it is not what the two-phase rule's gradient guarantee
covers (the keystone holds within a basin). It is, however, exactly Experiment C Arm 1's question — can a
local rule learn *with* designed bistability — and B3 supplies its first data point: the two-phase rule does
sculpt the wells it sits in. The readouts at the lowest-energy minima of 1–2 lattice spacings also say these
configurations are folded lattices, outside any small-deformation elastic model's validity.

**Honest caveats.**
- The validation's distinct-minima count depends on its start distribution (x₀ + 6 Gaussian starts of scale
  0.1) and energy tolerance 1e-5; it is a lower bound on the number of minima.
- 400 epochs at β = 0.01 as in B; longer training or other β were not swept here.
- Rescue counts (up to 43 of 3200 solves at d = 0.5) mean Newton's warm-started path frequently reached a
  saddle or failed its line search in the multi-welled regime; settle() then returned a nearby minimum, so
  the trained trajectory itself contains some basin changes that are not the physics of a quasi-static
  lattice. In a real material the equivalent is a snap.
- Readouts of 1–2 lattice spacings at the lowest-energy minima indicate self-overlapping lattices; the model
  has no contact or self-avoidance.

**Portfolio implications.** (1) `01`-B's parity foreclosure stands with both legs now measured, and the
commons' `expressivity-multistability-pincer` lesson can be re-cited in this form: *representability
foreclosure at small deformation; hysteretic (basin-selection) separation, not a lattice property, at large
deformation.* (2) Lesson candidate: **validate any positive separation on a nonlinear substrate by multi-start
minimization — a trained trajectory's readouts are basin-dependent once the landscape is multi-welled.**
(3) Experiment C Arm 1 gets its motivating data point and its first design constraint: a protocol-free rule
must either keep the lattice monostable (then parity is unrepresentable) or make the basin selection itself
the computation.
