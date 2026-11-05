# Status & roadmap — 01 · Coupled-Learning Networks (restart, 2026-11-05)

**Purpose:** an honest evaluation of where the project stands after ~4.5 months idle (last commit
2026-06-19), what reproduced, what did not, what the portfolio commons has since asked of this project,
and a ranked research roadmap. Written to get the project back off the ground; findings produced today
are in `2026-11-05-experiment-b2-findings.md`.

> Method discipline: **Observed** = measured today in this environment; **Evaluation** = judgement.

---

## Observed — reproduction audit

### Environment
Python 3.13.18, numpy 2.5.3, scipy (now used — see B2), pytest. Container had no pytest installed; the
declared `requirements.txt` was sufficient. The portfolio commons (`../00-shared-harness`) was **not
present** beside the repo; it was cloned read-only from `bshepp/physical-learning-commons` (public).

### Test suite
44/44 tests passed on the inherited tree (28.5 s). After today's vectorization, 44/44 in 9.3 s; after
today's additions, **53/53 in 11.8 s**.

### Experiment A — reproduces exactly
`python experiment_a.py` reproduced the recorded table to every printed digit (allostery 3.6736e-13 /
4.4890e-08, regression 1.2610e-04 / 9.2315e-04, sign 1.00, ρ 1.000, both PASS, audit 1.0).

### Experiment B — reproduces, with one knife-edge
`python experiment_b.py` → **MULTISTABILITY-FORECLOSE**, identical per-seed `local_sep` and `jump_rate`.
One discrepancy: the seed-1 **baseline** separation.

| build | seed-1 baseline_sep |
|---|---|
| findings doc (2026-06-19) | −5.9e-02 |
| inherited loop kernels, today | **−inf** (a final-evaluation solve raised) |
| vectorized kernels, today | −5.9461e-02 |

The only difference between the two builds today is floating-point summation order. A borderline free
solve (indefinite tangent stiffness after 400 GD steps) sits on a numerical knife-edge. The headline verdict
is unaffected (it is set by seed 2's jump rate 0.26 > 0.20), but it means **Experiment B's baseline numbers
near the buckling boundary are not bit-reproducible across numpy builds**, and the −inf / finite flip is
exactly the "harness artifact" class the commons warns about. Recorded here; not patched over.

### Performance (today)
| | before | after |
|---|---|---|
| `energy` / `forces` / `tangent_stiffness` (4×4 lattice) | 297 / 683 / 1668 µs | 29 / 48 / 139 µs |
| full suite | 28.5 s | 9.3 s |
| `experiment_b.py` | 4 m 14 s | 54 s |

All finite-difference gates (force, Hessian, adjoint-vs-FD for both `k` and `ℓ`) pass unchanged.

### GitHub
No open issues or pull requests. Branches: `master` (= last work) and `claude/upbeat-hamilton-85zr3l`
(this session's work).

---

## Evaluation — what the project is, and where it is weak

**What is solid.** The thesis test is well-built: a shared two-phase harness, a hand-written adjoint
cross-checked against finite differences, a provenance audit the smoke tests enforce, and pre-registered
verdicts with Observed/Interpretation kept separate. Experiment A's PASS is clean (and near-foregone, as
the spec says). Experiment B's *mechanism* result — the mechanical local rule equals the true gradient
as β→0 on a nonlinear elastic substrate — is a genuine, reusable keystone, and the commons has already
promoted the whole family to Tier 1.

**Where Experiment B's FORECLOSE is weaker than it reads.** The verdict has two legs; the multistability
leg is well-measured, but the *representability* leg rested on an offline baseline with three
unexamined choices, each of which the portfolio has since learned can manufacture a false foreclosure:

1. **Objective presupposed a scale.** The baseline minimized MSE to targets `±0.5·d` (= ±0.05 at
   d=0.1). The untrained lattice's parity-channel response at d=0.1 is ~5e-4 (measured today, see B2) —
   a hundred times smaller. An optimizer asked to hit an unreachable magnitude can settle on a
   non-separating compromise even if a separating configuration exists at a smaller scale. Two-sided
   separation is scale-free; the probe objective was not.
2. **Optimizer and budget were fixed.** Plain gradient descent, lr 1.0, 400 epochs, with failed epochs
   silently skipped (`continue`). No saturation evidence. This is the commons lesson
   `foreclosure-can-be-a-budget-artifact` verbatim (03's parity-4 "foreclosure" dissolved at 3000 steps).
3. **Alignment was measured at convergence.** `_alignment` ran on the trained weights; at a loss minimum
   both Δθ and ∇L → 0 and ρ is meaningless (seed 0 reported sign 0.00 / ρ 0.000). Commons pattern
   `alignment-certification`: measure at init. Fixed today (init is now the verdict input; final is kept
   as a secondary column).

None of this says the foreclosure is wrong — the d³ scaling measured today argues it is right — but it
means the claim was under-supported. **Experiment B2 (today) re-probes exactly these three points.**

**What the commons asked of `01` since June (BULLETIN, 2026-07-15):**
- Record the audit on **two axes** (gradient source × write mechanism); the canonical recorder is
  `commons/audit.py`, with `01`'s one-axis `Auditor` kept as a shim pending a RECONCILE. → *Roadmap item.*
- Before trusting a contrastive update as the gradient, confirm the relaxation **descends an energy**
  (`gradient-flow-is-the-ep-license`). Both of `01`'s substrates do (co-content / elastic energy) — this is
  why the keystones hit ρ→1; stated now in the README.
- Treat β as task-scaled and **report η/β**. → Done today for B (η/β = 100).
- The H5 read-noise sweep on `01`'s flow rule was **run by `07`**: the two-read rule is read-noise-fragile
  (FORECLOSE, provisional). Nothing for `01` to run there; the hardware lead (printed flow) is redirected
  toward a **fast/slow two-timescale directed-aging medium** that reconstructs the free phase in-fabric
  (H4) — squarely `01`-B elastic territory.
- `02`'s **two-ceiling** menu (`learnability_ceiling` substrate-independent MLP; `representability_ceiling`
  with L-BFGS or DE, scoring MSE or two-sided separation) is Tier 1 and the recommended representability
  probe. B2 implements the same idea natively (scipy is already a declared dependency here).

---

## Roadmap (ranked)

1. **B2 — representability re-probe (done today; see findings).** Scale-free Walsh-purity objective,
   L-BFGS-B with saturation trace, MSE-under-L-BFGS control, DE optimizer-class check, untrained Walsh
   spectrum vs d. Outcome recorded in `2026-11-05-experiment-b2-findings.md`.
2. **Two-axis audit adoption.** Port `commons/audit.py`'s `Audit` (keep `AuditLog` as the shim so the
   smoke tests' `fraction_physics_local == 1.0` invariant is untouched), and tag headline updates
   `(physical, physical)`, baselines `(cpu, electronic)`. Small, mechanical, closes the open RECONCILE.
3. **B's multistability leg: a monostable-regime learnability study.** The B design said a
   MULTISTABILITY-FORECLOSE "would motivate damped-dynamics relaxation or a monostable-regime study".
   With kernels 10× faster, sweep (d, pre-stress) with a jump-rate map and ask whether *any* region is
   both stable and learnable for a friendlier mechanical task (allostery-in-elastic) — this separates
   "the rule breaks" from "parity is hard" on the elastic medium.
4. **Experiment C — fast/slow two-timescale elastic medium (H4).** The commons' diagnosed next
   build-direction for the compliant/directed-aging hardware route: a fast mode physically plays the
   free-phase reference each step, a slow mode is the aged weight. Simulate it on the spring network;
   pre-register PASS/FORECLOSE on allostery first (friendly), parity later. This is the most novel open
   question `01` can own, and it feeds `07`'s hardware arm directly.
5. **A — show the β tradeoff on a heterogeneous net; exercise the Poiseuille law.** Minor; A's findings
   flagged the β-dependence as barely visible, and `poiseuille_law` is implemented but never run in a
   headline. Cheap completeness items.
6. **Read-noise on the elastic rule (H5 for B).** Only worth running if (3) or (4) finds a learnable
   regime; a noise sweep on a foreclosed substrate is uninformative.

---

## Files touched today
`spring_network.py` (vectorized), `baselines.py` (`mech_readout_adjoint`, vectorized), `parity_analysis.py`
(new), `experiment_b2.py` (new), `tasks.py` (`verdict_b2`), `experiment_b.py` (alignment at init, η/β),
tests for each. Scratch outputs (`*_results.json`) remain gitignored.
