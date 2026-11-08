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

## Correction (2026-11-06) — a solver artifact in the crash-tolerant seam

Experiment B0's trajectory diagnostics traced "instability events" to the free-phase Newton solve
**cold-started from the reference geometry** failing its line search on pre-stressed lattices whose
equilibria are unique and stable. The seam now warm-starts from the previous equilibrium, falls back to a
Levenberg–Marquardt minimizer (`settle()`), counts rescues separately, and **measures** the stability margin
(smallest tangent-stiffness eigenvalue). Consequences, all re-run and recorded in addenda:
- **Experiment B:** jump rates 0.19/0.00/0.26 → **0.00/0.00/0.00**; verdict MULTISTABILITY-FORECLOSE →
  **FORECLOSE-REPRESENTABILITY**. The "pincer" multistability leg is withdrawn; the representability leg
  (confirmed by B2) is the whole verdict. The commons' `expressivity-multistability-pincer` lesson should be
  re-read with this.
- **Experiment B0:** amplify MULTISTABILITY-FORECLOSE → **AMBIGUOUS** (2 PASS, 1 learns-but-misses-GD);
  "loss descent marches toward buckling" withdrawn; stability margin drifts 10–30 % over 6000 epochs.
- **Read-noise sweep:** "tax paid as buckling" → drift; β-scaling of the wall now resolved (1e-3 → 1e-2).
- **Experiment B2:** verdict unchanged (FORECLOSE-REPRESENTABILITY-CONFIRMED, 42/42 negative); eval-fail
  rates 32–47 % → 0 %, "wall-pinned" 1.00 → 0–0.10 (withdrawn); purity reached 0.70 and a₁₂₃ 1.5 with
  separation still negative — a representability wall, not an instability wall. Untrained lattice stable
  through d = 0.5.
- **Experiment B3 (new):** the multistability leg re-established with measured minima — hysteretic
  separation at d ≥ 0.3, not a lattice property; FORECLOSE-REPRESENTABILITY at d ≤ 0.2.
- **Lesson candidates for the commons:** (i) a crash-tolerant solver seam must distinguish "the solver failed
  from its start point" from "no stable equilibrium exists"; count only the latter as physics, and *measure*
  stability (smallest Hessian eigenvalue); (ii) validate any positive separation on a nonlinear substrate by
  multi-start minimization — a trained trajectory's readouts are basin-dependent once the landscape is
  multi-welled.

## Roadmap (ranked)

1. **B2 — representability re-probe (done today).** Outcome **FORECLOSE-REPRESENTABILITY-CONFIRMED**:
   parity channel ∝ d³ vs linear ∝ d (crossover far beyond buckling); optimizing (k, ℓ) grows the parity
   coefficient ×600 but every optimum is wall-pinned and 38/38 separations are negative. B's verdict
   stands on a budget-saturated, scale-free, two-optimizer probe. See
   `2026-11-05-experiment-b2-findings.md`.
2. **Two-axis audit adoption** — *done 2026-11-05* (`audit.py` projects onto the commons' axes behind the
   existing API). Port `commons/audit.py`'s `Audit` (keep `AuditLog` as the shim so the
   smoke tests' `fraction_physics_local == 1.0` invariant is untouched), and tag headline updates
   `(physical, physical)`, baselines `(cpu, electronic)`. Small, mechanical, closes the open RECONCILE.
3. **B's multistability leg: a monostable-regime learnability study** — *done as Experiment B0,
   2026-11-06, and it dissolved the leg*: attenuate PASS, invert PASS (a sign inversion, 100 % physics-local),
   amplify AMBIGUOUS (2 PASS; one seed 2.6× behind matched GD); no instability at d_max ≤ 0.3; the rule
   beats matched-budget GD in 7/9 cells. Read-noise sweep (item 6): H5-MECHANISM-CONFIRMED with β-scaling.
   See the two 2026-11-06 findings docs and the correction above.** The B design said a
   MULTISTABILITY-FORECLOSE "would motivate damped-dynamics relaxation or a monostable-regime study".
   With kernels 10× faster, sweep (d, pre-stress) with a jump-rate map and ask whether *any* region is
   both stable and learnable for a friendlier mechanical task (allostery-in-elastic) — this separates
   "the rule breaks" from "parity is hard" on the elastic medium.
4. **Experiment C** — *design draft written 2026-11-06* (`2026-11-06-experiment-c-design.md`); first
   blocker closed by `mechanical_solver.settle()`; **first-step measurement done 2026-11-08**: on a 4-node
   snap device the two-phase rule learns a bistable switch by folding the wrong well away and snapping the
   state to the right one (ρ ≈ 1 in both wells, ~0.5 near the fold). Arm 1 is worth a full plan. **Fast/slow two-timescale
   elastic medium (H4).** *(B2 adds a second candidate C:
   elements bistable by design, with a rule that tolerates multistability — since B2 shows the
   parity-increasing direction IS the buckling direction, an escape on this medium must embrace
   bistability rather than avoid it.)* The commons' diagnosed next
   build-direction for the compliant/directed-aging hardware route: a fast mode physically plays the
   free-phase reference each step, a slow mode is the aged weight. Simulate it on the spring network;
   pre-register PASS/FORECLOSE on allostery first (friendly), parity later. This is the most novel open
   question `01` can own, and it feeds `07`'s hardware arm directly.
5. **A — β tradeoff + Poiseuille** — *done as Experiment A2, 2026-11-06*: 1 − ρ ∝ β^2.00, Poiseuille PASS. Minor; A's findings
   flagged the β-dependence as barely visible, and `poiseuille_law` is implemented but never run in a
   headline. Cheap completeness items.
6. **Read-noise on the elastic rule (H5 for B)** — *done 2026-11-06* (see item 3). Open: resolve the
   β-scaling of the independent wall with a half-decade σ grid. Only worth running if (3) or (4) finds a learnable
   regime; a noise sweep on a foreclosed substrate is uninformative.

---

7. **(Downgraded.) Stability-regularized local rule.** Motivated by B0 run 1's buckling, which was the
   solver artifact. The measured margin drifts only 10–30 % over 6000 epochs; revisit only if a longer
   budget or larger d_max shows the margin actually approaching zero (`min_eig_history` now records it).
8. **Conditioning of the two-phase update vs plain GD** — *done 2026-11-08* (B0 findings, "Conditioning
   follow-up"): loss Hessian λ_max 2.09 (rest lengths) vs 5e-4 (stiffnesses); GD stability bound lr < 0.96,
   so matched GD at lr 1 was at its edge and spikes; the finite-β update turns away from the raw gradient
   along the trajectory (cosine 0.65, magnitude 0.28×) and never spikes — an implicitly damped step on stiff
   directions. Open: does it generalize (other tasks, the resistor net)?
9. **Read-noise follow-ups.** β-scaling is resolved (run 2). Open: the *noise-helps transient* — at σ = 1e-3
   the noisy trajectory's best error is ~100× below the clean run's; pre-register an annealed-σ run
   (commons `noise-as-resource`).
10. **Experiment B's multistability question, done properly** — *done as Experiment B3, 2026-11-06*:
    unique minima and FORECLOSE-REPRESENTABILITY at d ≤ 0.2; at d ≥ 0.3 the lattice is multi-welled (2–5
    minima per pattern), the local rule's readouts separate parity-3 (3/3 seeds at d = 0.5) but only as
    basin selection — lowest-energy minima never separate (MULTISTABLE-HYSTERETIC). The pincer stands in
    measured form. Next: Experiment C Arm 1 (make basin selection the computation, or keep it out).

## Session log — what was done (2026-11-05 → 11-06), in order

1. Reproduction audit (A exact; B reproduces; seed-1 knife-edge noted). Spring kernels vectorized (10–14×).
2. **B2** — representability re-probe: FORECLOSE-REPRESENTABILITY-CONFIRMED (run 2; run 1 apparatus-limited).
3. Two-axis audit projection; alignment certified at init; η/β reported.
4. **B0** — mechanical allostery (the missing positive control): attenuate PASS, invert PASS, amplify
   first read MULTISTABILITY-FORECLOSE …
5. … whose diagnosis found the **cold-start solver artifact**. Seam fixed (warm start + `settle()` fallback,
   saddle rejection, measured stability margin). Re-runs: **B → FORECLOSE-REPRESENTABILITY (jump 0.00)**,
   **B0 amplify → AMBIGUOUS (2 PASS)**, **B2 verdict unchanged, "wall-pinned" withdrawn**, **noise sweep:
   β-scaling resolved, drift not buckling**.
6. **B3** — multistability measured: FORECLOSE-REPRESENTABILITY (d ≤ 0.2), MULTISTABLE-HYSTERETIC (d ≥ 0.3).
7. **A2** — 1 − ρ ∝ β^2.00; Poiseuille PASS. **Read-noise** — H5-MECHANISM-CONFIRMED (+ noise-helps transient).
8. **Experiment C** design draft; `settle()` built and shown to follow a snap with hysteresis.

Every verdict change is recorded as an addendum beside the original numbers; nothing was overwritten.

## Files touched (both days)
Source: `spring_network.py` (vectorized), `mechanical_solver.py` (`min_eig`, `settle()`), `mechanical_learning.py`
(continuous targets, trajectory diagnostics, read noise, robust free phase), `baselines.py`
(`mech_readout_adjoint`, `mech_ceiling`, significant sign-match), `tasks.py` (`allostery_mech_task`,
`verdict_b0/_b0_noise/_b2/_b3`, `verdict` abs_floor), `audit.py` (two-axis), `experiment_b.py` (init
alignment, η/β, margin), new `parity_analysis.py`, `experiment_a2.py`, `experiment_b0.py`,
`experiment_b0_noise.py`, `experiment_b2.py`, `experiment_b3.py`; one test module per source module (76 tests).
Docs: this file; findings for A2, B0, B0-noise, B2, B3; addenda in B's findings; Experiment C design;
README and CLAUDE.md. Scratch outputs (`*_results.json`) remain gitignored; run logs are in the session
scratchpad only.
