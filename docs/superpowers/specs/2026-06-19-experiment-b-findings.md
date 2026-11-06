# Findings — Experiment B: signed/non-monotone elastic networks vs parity-3

**Date:** 2026-06-19
**Verdict:** **FORECLOSE.** Signed/non-monotone elastic state does **not** clear the parity-3 wall in this
substrate. It manifests two ways: **representability foreclosure** in the stable (low-deformation) regime
and **multistability foreclosure** in the non-monotone (high-deformation / random-init) regime. The
coupled-learning *mechanism* is sound — the limitation is the substrate, not the learning rule.
**Implication:** the wall is deeper than passivity → escalate to project `03` (energy-based/Ising), exactly
the SEED's pre-registered FORECLOSE-B path.

> Method discipline: **Observed** = measured numbers; **Interpretation** kept separate. Judged against the
> pre-registered verdicts in `2026-06-19-experiment-b-design.md`.

---

## Observed

### Test suite
44 tests passing (`python -m pytest -q`), one module per source module. Numerical gates: force `= −∂E/∂x`
and tangent stiffness `= ∂²E/∂x²` validated against finite differences (symmetric and asymmetric `α`);
lattice rigidity (`H_ff` full-rank after pinning); the implicit adjoint validated against an independent
central-difference gradient for **both `k` and `ℓ`** (max error ~1e-9).

### Keystone — the mechanism is correct
`test_local_update_aligns_with_true_gradient_as_beta_shrinks`: the local two-phase `(Δk, Δℓ)` update
converges to the negative true (adjoint) gradient as the nudge `β→0`.

| β | ρ (local update vs −true gradient) | sign-match |
|------|------------------------------------|-----------|
| 5.0   | 0.836 | 0.53 |
| 1.0   | 0.988 | — |
| 0.1   | 0.99993 | — |
| 0.01  | 1.0000 | — |
| 0.002 | **1.0000** | **1.00** |

The mechanical coupled-learning rule **is** the loss gradient in the small-nudge limit (the
equilibrium-propagation result, now confirmed on a nonlinear elastic substrate). Independently of parity,
this extends Experiment A's "the physics computes the update" to the signed/elastic medium.

### Headline experiment (default config, randomized per-seed init, `d=0.1`, `α=1`, 4×4 lattice)

| seed | local_sep | baseline_sep | jump_rate | sign | ρ |
|------|-----------|--------------|-----------|------|-----|
| 0 | −8.1e-02 | −inf (unstable) | 0.19 | 0.00 | 0.000 |
| 1 | −6.2e-02 | −5.9e-02 | 0.00 | 0.97 | 1.000 |
| 2 | −1.3e-01 | −inf (unstable) | 0.26 | 0.97 | 1.000 |

**Verdict: MULTISTABILITY-FORECLOSE** (worst-seed jump_rate 0.26 > 0.20 threshold; audit physics-local =
1.00). Random initial spring configurations drift into indefinite-tangent-stiffness (snap-through) regions
during training, where the two-phase rule is ill-defined.

### Representability characterization (offline optimizer — the deeper, more robust probe)

The offline adjoint-gradient baseline's **best** achievable two-sided parity separation (positive = the
classes are separable; the verdict needs > 0):

*d-sweep (deterministic init, 1 seed):*

| d | verdict | baseline_sep | jump_rate |
|------|---------|--------------|-----------|
| 0.05 | FORECLOSE-REPRESENTABILITY | −8.2e-02 | 0.00 |
| 0.10 | FORECLOSE-REPRESENTABILITY | −1.6e-01 | 0.00 |
| 0.15 | FORECLOSE-REPRESENTABILITY | −1.9e-01 | 0.00 |
| 0.20 | FORECLOSE-REPRESENTABILITY | −inf (partial instability) | 0.17 |
| 0.25 | FORECLOSE-REPRESENTABILITY | −inf (partial instability) | 0.10 |
| 0.30 | MULTISTABILITY-FORECLOSE | −inf | 0.21 |

*randomized-init baseline (best parity_sep over 6–8 random inits):*

| lever swept | values | best baseline parity_sep |
|---|---|---|
| init seed (d=0.10) | 8 inits | −1.4e-01 |
| init seed (d=0.15) | 8 inits | −1.4e-01 |
| asymmetry `α` (d=0.15) | 1.0 / 0.7 / 0.5 / 0.3 | −0.14 / −0.16 / −0.12 / −0.11 |
| network size (d=0.15) | 4×4 / 5×5 / 6×6 | −0.16 / −inf / −0.26 |

**Across every swept lever — deformation `d`, asymmetry `α`, network size, and randomized initialization —
the best achievable parity-3 separation is negative.** The substrate cannot *represent* parity-3 anywhere in
the stable regime.

---

## Interpretation

**Signed/non-monotone elastic state does not clear the parity-3 wall here — a clean FORECLOSE.** The result
has a precise two-part structure that is itself the scientific content:

1. **Stable regime → representability foreclosure.** Where the energy landscape is single-basin (low `d`),
   the elastic network — even optimized offline from many random initializations, across asymmetry settings
   and network sizes — cannot achieve positive parity-3 separation. Geometric nonlinearity at small
   deformation is too weak to produce the non-monotone response parity requires; the tension–compression
   asymmetry lever did not rescue it (separation stayed negative for all `α`).

2. **Non-monotone regime → multistability foreclosure.** The deformation magnitude needed to make the
   response strongly non-monotone is exactly where the network buckles: the tangent stiffness goes
   indefinite, the equilibrium becomes multistable, and the free/nudged phases can no longer be relied upon
   to share a basin. The two-phase rule is ill-defined there. The headline multi-seed run lands here
   (MULTISTABILITY-FORECLOSE) because random initial configurations reach snap-through during training.

**These two regimes pincer the hypothesis:** the regime expressive enough to *possibly* separate parity is
the regime where the learning rule breaks down, and the regime where the rule is well-defined is not
expressive enough. That tension — anticipated in the spec's multistability risk — is the substantive finding.

**The mechanism is not implicated.** The keystone shows the local coupled-learning update equals the true
gradient (ρ=1.0, sign-match 1.0 as `β→0`); where the solve is stable, gradient alignment is near-perfect.
So this is not a failure of co-located learning — it is a property of the *passive elastic substrate*. The
audit confirms every headline weight update was physics-local (fraction = 1.00): the physics still computes
the update; the substrate simply cannot host parity-3.

**Honest caveats.**
- Representability was probed with **gradient descent (a local optimizer)** from random inits, not an
  exhaustive/global search. A radically different topology, much larger network, or a global optimizer
  *might* find a separating configuration we did not. The claim is "strongly unsupported across a broad,
  reasonable sweep," not "provably impossible."
- The representability-vs-multistability boundary is sensitive to init and hyperparameters (the default
  config reports MULTISTABILITY-FORECLOSE; the deterministic/stable-regime probes report REPRESENTABILITY
  foreclosure). Both are FORECLOSE; the distinction is *why*, not *whether*.
- All runs used the 2D triangulated lattice with one I/O assignment. Topology/I/O placement were not swept.

**Portfolio implication.** This is the SEED's pre-registered **FORECLOSE (B)**: even signed, non-monotone
elastic state can't separate parity-3 within physical limits → the wall is deeper than passivity. The next
move is **project `03` (energy-based/Ising)**, whose non-monotone-by-construction state (discrete spins) is
not subject to the buckling/multistability tradeoff that pincers the elastic substrate. Experiment A (PASS,
the mechanism is real) and Experiment B (FORECLOSE, the elastic medium can't host parity) together
characterize the boundary: co-located learning works; clearing the parity wall needs a substrate whose
non-monotonicity does not come with a multistability penalty.

---

## Addendum (2026-11-06) — the multistability leg was a solver artifact; the verdict is FORECLOSE-REPRESENTABILITY

**What was found.** Experiment B0 (mechanical allostery, 2026-11-06) traced an "instability event" during
training to its source: the free-phase Newton solve, **cold-started from the reference geometry `x0`** on a
lattice whose rest lengths had drifted far from geometry, failed its line search — while the equilibrium it
was looking for is **unique** (eight random starts converge to one minimum) and **stable** (smallest
tangent-stiffness eigenvalue +0.028). A Levenberg–Marquardt-damped solver (`mechanical_solver.settle`) or a
Newton warm-started from a neighbouring equilibrium reaches it in 2–8 iterations. The `RuntimeError` that
this experiment counts as a buckling/multistability event was therefore, at least in these cases, a
**cold-start artifact of the Newton line search**, not a physical event.

**The fix (commit "free phase warm-starts and falls back to settle()").** `free_phase_mech` now warm-starts
each pattern from its previous equilibrium and falls back to `settle()` when Newton fails, flagging the
rescue; only a failure of `settle()` too is counted as instability. `relax_mech` itself is unchanged.

**Re-run of this experiment under the corrected seam** (`python experiment_b.py`, default config, identical
seeds and inits):

| seed | local_sep | baseline_sep | jump_rate | sign@init | ρ@init | ρ@end |
|---|---|---|---|---|---|---|
| 0 | −7.20e-02 | −7.34e-02 | **0.00** | 0.97 | 1.000 | 1.000 |
| 1 | −6.17e-02 | −5.95e-02 | 0.00 | 0.95 | 1.000 | 1.000 |
| 2 | −7.32e-02 | −4.40e-02 | **0.00** | 0.98 | 1.000 | 1.000 |

**Verdict under the corrected seam: FORECLOSE-REPRESENTABILITY** (SNR −73; audit 1.00; η/β = 100). Every
jump rate is zero; every baseline is finite; seed 0's ρ@end reads 1.000 (the June 0.000 was the same
artifact, caught at the alignment step). Compare the June table: jump rates 0.19 / 0.00 / 0.26 and two
`−inf` baselines — all of them cold-start failures.

**What changes in the interpretation.**
- The **two-part "pincer"** as stated above is **not supported by this experiment's data**. The
  *representability* leg stands and is now the whole verdict (and is independently confirmed and
  quantified by Experiment B2: parity channel ∝ d³, negative separation in 38/38 offline optima). The
  *multistability* leg — "the regime non-monotone enough to represent parity is where the lattice buckles
  and the rule becomes ill-defined" — was inferred from instability counts that this seam produced
  spuriously at d = 0.1. Whether genuine multistability appears at larger deformation is a separate, open
  question (B2's re-run under the corrected seam addresses the untrained lattice's stability at d = 0.3).
- The learning *mechanism* conclusion is unchanged and stronger: ρ = 1.000 at init on every seed.
- The portfolio statement "the wall is deeper than passivity" survives on the representability leg alone.
- For the commons: `lessons-ledger#expressivity-multistability-pincer` should be read with this addendum;
  the multistability half of the pincer needs re-establishing with the corrected seam before it is cited.
  A new lesson candidate replaces it: **a crash-tolerant solver seam must distinguish "the solver failed
  from its start point" from "no stable equilibrium exists"** — count only the latter as physics. The
  original design (§ "Crash-tolerance is intentional") conflated them.

**What does not change.** The June table above is left as recorded (that is what the code of the day
produced); this addendum is the correction. Observed numbers for the corrected seam are in
`experiment_b_results.json` from the 2026-11-06 run and in `2026-11-06-experiment-b0-findings.md`.
