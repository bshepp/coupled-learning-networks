# Design — Experiment B: signed/non-monotone elastic networks vs parity-3

**Date:** 2026-06-19
**Status:** DESIGN (approved; next step `writing-plans`).
**Project:** `coupled-learning-networks` (portfolio project 01). Builds on the harness validated by
**Experiment A** (`2026-06-18-coupled-learning-networks-design.md`, `2026-06-18-experiment-a-findings.md`).
**Repo:** same repo as A (Approach 1 — parallel mechanical modules; A is not modified, only `tasks.py`
and `baselines.py` are extended additively).

---

## 1. Scope & the question

Experiment B tests whether a **signed, non-monotone elastic substrate** can achieve a parity-3 separation
that passive positive-conductance networks provably cannot — i.e. whether *the parity wall is a property
of passive monotone substrates, not of co-located learning itself.*

A established (PASS) that the local, physics-driven coupled-learning rule reaches the offline baseline and
matches the true gradient — *the physics computes the update* — but on a passive resistor network the
maximum principle makes the output monotone in the inputs, foreclosing parity by construction. B replaces
the substrate with a **2D spring network in the finite-deformation regime**, where geometric nonlinearity
(rotating bond directions) makes the input→output map genuinely non-monotone, and asks the SEED's
pre-registered question:

- **PASS (B):** a feasible, robust positive parity-3 separation in the elastic network → *the wall is about
  passivity, not co-located learning.*
- **FORECLOSE (B):** even signed elastic state can't separate parity-3 within physical limits → *the wall is
  deeper than passivity* (escalate to active/energy-based, project `03`).

**Honest framing.** B is the genuine scientific bet of project 01 (A's PASS was near-foregone). The result
is **unknown** — a clean FORECLOSE is as valid as a PASS. The design is built so that a FORECLOSE cannot be
dismissed as under-powered (maximum expressivity: learn both `k` and `ℓ`) and so that a *third* outcome —
the learning rule breaking down under multistability — is detected and reported distinctly rather than
masquerading as a foreclosure.

---

## 2. Architecture (Approach 1 — parallel mechanical modules)

New flat modules beside A's; reuse A's audit/verdict/alignment surface; A's files untouched (`tasks.py`,
`baselines.py` extended additively). House code pattern throughout (small modules, functions return
numbers, runnable modules have a `main()` that prints a table + dumps JSON, one pytest module per source
module).

| Module | Role | New / reuse |
|---|---|---|
| `spring_network.py` | `SpringNetwork` (2D positions, edges, rest lengths `ℓ`, stiffness `k`, pinned DOFs, input nodes + displacement encoding, output node + readout component); spring **energy / force / tangent-stiffness** with smoothed tension–compression asymmetry `m(e)`; triangulated-lattice builder | new |
| `mechanical_solver.py` | `relax_mech(...)` — damped Newton energy-minimization with line search; warm-start; `basin_jump(...)` detector | new |
| `mechanical_learning.py` | `free_phase_mech`, `nudged_phase_mech` (warm-started), `local_update_mech` (edge-local `Δk`, `Δℓ`), `train_mech`; basin-jump rate; records `PHYSICS_LOCAL` | new |
| `tasks.py` | **+** `parity3_mech_task`, `two_sided_separation`, `verdict_b` | extend |
| `baselines.py` | **+** `mech_adjoint_gradient`; reuse `gradient_alignment` | extend |
| `experiment_b.py` | driver: train (audited) → adjoint-GD baseline → separation/SNR + alignment + basin-jump rate → verdict + JSON | new |
| `audit.py` | `AuditLog`, provenance constants | reuse as-is |

Tests: `test_spring_network.py`, `test_mechanical_solver.py`, `test_mechanical_learning.py`, extended
`test_baselines.py` / `test_tasks.py`, `test_experiment_b.py`.

The `relax_mech` seam mirrors A's `relax` (clamp → relax → state); the conceptual structure carries over
even though the state is vector positions, not scalar voltages. (The canonical `00` harness extraction —
unifying A's `relax` and B's `relax_mech` — is deferred to the later cross-portfolio reconciliation, per
the portfolio README.)

---

## 3. Spring network model & constitutive law

**State:** 2D node positions `x ∈ ℝ^(N×2)` (2 DOFs/node). Bonds `(i,j)` carry stiffness `k_ij > 0` and rest
length `ℓ_ij > 0`.

**Energy / force (geometric nonlinearity):** bond energy `E_ij = ½ k_ij · m(e_ij) · e_ij²`, where
`e_ij = L_ij − ℓ_ij`, `L_ij = |x_i − x_j|`. Total `E = Σ E_ij`; `F_i = −∂E/∂x_i`, magnitude `k·m(e)·e` along
the unit bond vector `d̂ = (x_i−x_j)/L_ij`. The geometric nonlinearity (through `L` and the rotating `d̂`)
is the source of non-monotonicity in the finite-deformation regime.

**Tension–compression asymmetry (smoothed lever):** `m(e) = α + (1−α)·½(1 + tanh(e/w))`, `α ∈ (0,1]`,
smoothing width `w`. At **`α = 1`, `m ≡ 1` → symmetric springs = the primary configuration**; `α < 1` is the
asymmetry lever (stiffer in tension). Smoothness keeps `E`, `F`, `H` in `C∞` so the implicit adjoint stays
valid.

**Rigidity & pinning:** a generic under-braced spring network has zero-energy floppy modes → singular
Hessian → degenerate solve. The builder uses a **2D triangulated lattice** (generically rigid). Rigid-body
modes are removed by pinning DOFs (fix one node fully + one DOF of a second → removes 2 translations + 1
rotation). Pinned DOFs and driven input DOFs are held fixed (Dirichlet) during relaxation.

**I/O encoding (imposed displacements):** 3 input nodes; bit `s_i ∈ {+1,−1}` → prescribed displacement
`s_i·d` of input node `i` along a fixed axis from its reference position. A distant output node's scalar
displacement component is the readout. **Deformation magnitude `d` is the key lever** balancing
non-monotonicity (large `d`) against multistability (too-large `d`).

---

## 4. The mechanical coupled-learning rule

Per parity pattern (8 patterns, full-batch), contrastive convention consistent with A
(`Δθ = (η/β)[∂E_free/∂θ − ∂E_nudged/∂θ]`, edge-local):

1. **Free phase** — clamp the 3 input DOFs to `±d`; `relax_mech` from reference → `x_free`; read `r_free`.
2. **Nudged phase** — keep inputs clamped; clamp the output readout DOF to `r_free + β(target_class − r_free)`;
   `relax_mech` **warm-started from `x_free`** → `x_nudged`; call `basin_jump(x_free, x_nudged, β)` and
   accumulate the rate.
3. **Local update** (per bond, using only that bond's own `e = L−ℓ` and `k`):
   - `Δk_ij = (η_k/β)·[ ½ m(e_free)e_free² − ½ m(e_nudged)e_nudged² ]`  (`∂E/∂k = ½ m(e)e²`; symmetric →
     `(η_k/2β)(e_free² − e_nudged²)`).
   - `Δℓ_ij = (η_ℓ/β)·[ ∂E_free/∂ℓ − ∂E_nudged/∂ℓ ]`, with `∂E/∂ℓ = −k[ ½ m'(e)e² + m(e)e ]` (symmetric →
     `Δℓ_ij = (η_ℓ k/β)(e_nudged − e_free)`).

Both updates use only local bond quantities — the audit point holds as in A; `local_update_mech` records
`PHYSICS_LOCAL`. **Sign is verified, not asserted** — the gradient-alignment test (§6) checks alignment with
`−∂L/∂θ` as `β→0`.

**Bounds:** clip `k ∈ [k_min, k_max]` (`k_min > 0`) and `ℓ ∈ [ℓ_min, ℓ_max]` (`ℓ_min > 0`) every step.
**Note:** unlike A, `k > 0` does *not* impose monotonicity — the geometric nonlinearity supplies the
non-monotone response, which is exactly why B can attempt parity while A cannot.

`train_mech` loops epochs over the 8 patterns, updates `(k, ℓ)`, clips, and tracks per-epoch separation and
the cumulative basin-jump rate.

---

## 5. Parity-3 task, separation, and verdict

**`parity3_mech_task`:** 8 patterns `s ∈ {±1}³`, label `y = s₁·s₂·s₃`; a triangulated lattice with 3 input
nodes, 1 distant output node, pinned nodes; input encoding `s_i → ±d`; two per-class readout targets
(`target_+`, `target_−`) spanning the achievable readout range.

**`two_sided_separation(r, y)`** (ported from `self-assembly/stage1_network.py`):
`max( min(r|y=+1) − max(r|y=−1), min(r|y=−1) − max(r|y=+1) )`. `> 0` iff a threshold separates the classes
in *either* polarity; `≤ 0` iff they overlap. An **SNR** is formed against a readout-noise floor (stage1
idiom) so PASS is not declared on a hair-thin margin.

**`verdict_b`** — evaluated in this order:

| Verdict | Condition | Meaning |
|---|---|---|
| **MULTISTABILITY-FORECLOSE** | basin-jump rate during training > threshold | the two-phase rule is ill-defined here — *checked first*; distinct from any separation result |
| **FORECLOSE (representability)** | offline-baseline separation ≤ 0 (SNR < 1) | even the optimizer can't separate parity-3 within physical limits → the wall is deeper than passivity |
| **FORECLOSE (learnability)** | baseline separates but the local rule can't | representable but not locally learnable |
| **PASS** | local-rule separation SNR ≥ k, **robust across seeds**, low basin-jump rate, reasonable gradient alignment | signed/non-monotone elastic clears the wall, and the physics learned it |
| **AMBIGUOUS** | positive but sub-threshold / fragile across seeds | marginal |

**Robustness is required for PASS:** positive separation must hold across multiple seeds (random
lattices/init), not one lucky configuration (per the SEED's "feasible, robust" wording and A's warning about
fragile single-seed results). Thresholds (`k`, basin-jump rate, seed count) are config constants.

---

## 6. Baseline (implicit-adjoint gradient) & gradient alignment

**Optimizer/alignment loss:** differentiable per-class target MSE `L = ½ Σ_patterns (r_p − target_{class(p)})²`
(drives each readout to its class target → positive separation; same output-vs-target form as A).

**`mech_adjoint_gradient(net, k, ℓ, task) → (dL/dk, dL/dℓ)`** — implicit function theorem through the
equilibrium. At `∇_x E = 0`, `dx_free/dθ = −H_ff⁻¹ ∂(∇_free E)/∂θ`, so `dL/dθ = −λ · ∂(∇_free E)/∂θ` with the
adjoint solve `H_ff λ = ∂L/∂x_free` (`H_ff` symmetric). **Reuses the Newton tangent-stiffness `H_ff` from
`relax_mech`** — computed once, used for solve and gradient. Returns gradients w.r.t. both `k` and `ℓ`.

**FD cross-check (test, the source of truth):** central differences — perturb each `k`/`ℓ`, re-solve,
recompute `L`, central-difference. Independent of the adjoint code path; the test asserts `adjoint ≈ FD`,
pinning the adjoint's correctness *and* sign (A's pattern). Valid for the smoothed-`m` (`α<1`) case since `E`
is `C∞`.

**`gd_baseline_mech`:** gradient descent on the adjoint gradient over `(k, ℓ)` to convergence → the
representability/learnability ceiling (best achievable separation), tagged `OFFLINE_GRADIENT`.

**`gradient_alignment` (reused from A):** compare the concatenated local update `(Δk, Δℓ)` to `−`adjoint
gradient — sign-match, ρ, cosine — across a β-sweep, expecting `→1` as `β→0` *when no basin-jump occurred*.
Doubles as a multistability diagnostic.

---

## 7. Testing strategy

TDD; one test module per source module; numpy + scipy + pytest; no new heavy deps (adjoint hand-written).

| Test module | Key checks |
|---|---|
| `test_spring_network.py` | force `= −∂E/∂x` and Hessian `= ∂force/∂x` vs finite differences; symmetric (`α=1`) and smoothed-`m` limits/`C¹`; triangulated builder rigid (`H` non-singular after pinning); I/O encoding correct |
| `test_mechanical_solver.py` | `relax_mech` hits a hand-solved equilibrium (single bond / triangle); damped Newton converges; warm-start reproduces the same equilibrium; `basin_jump` detects a constructed snap-through **and** returns False in-basin; floppy/singular net raises |
| `test_mechanical_learning.py` | free/nudged phases; nudge moves readout toward target; **keystone: `(Δk,Δℓ)` aligns with `−`adjoint gradient as `β→0`** (sign-match→1, ρ→1); `train_mech` reduces loss / raises separation on a representable case; audit 100% physics-local |
| `test_baselines.py` (extend) | `mech_adjoint_gradient` matches FD for **both `k` and `ℓ`**; `gd_baseline_mech` converges |
| `test_tasks.py` (extend) | `parity3_mech_task` structure; `two_sided_separation` (positive iff separable either polarity, ≤0 on overlap); all `verdict_b` branches |
| `test_experiment_b.py` | end-to-end smoke → a **valid** verdict dict (one of the five statuses) + JSON; audit physics-local; runs without basin-jump blowup |

**Critical distinction from A:** A's smoke test could assert `regression == PASS` (near-foregone). **B's
scientific outcome is unknown** — so `test_experiment_b.py` asserts only that the pipeline runs and emits a
*valid* verdict, **never a specific verdict**. The PASS/FORECLOSE is the experiment's *result* (run
`experiment_b.py`, record in a findings doc), not baked into a test.

---

## 8. Pre-registered verdicts (summary)

- **PASS:** robust (cross-seed) positive parity-3 separation, SNR ≥ k, low basin-jump rate, gradient
  alignment holds → the wall is about passivity, not co-located learning; the physics learned it.
- **FORECLOSE (representability):** even the offline optimizer can't separate parity-3 within physical
  limits → the wall is deeper than passivity → escalate to project `03` (energy-based/Ising).
- **FORECLOSE (learnability):** representable offline but the local rule can't learn it.
- **MULTISTABILITY-FORECLOSE:** the basin-jump rate is too high for the two-phase rule to be well-defined →
  a distinct result (the rule, not the substrate, is the binding constraint); would motivate damped-dynamics
  relaxation or a monostable-regime study.

---

## 9. Out of scope / deferred

- **Canonical `00` harness extraction** (unifying `relax` + `relax_mech` behind one interface) — deferred to
  the cross-portfolio reconciliation once `02`–`05` converge.
- **3D networks, dynamic/inertial effects, contact** — quasi-static 2D only.
- **Hardware** — compute/simulation first, per portfolio convention.

---

## 10. References

- Stern, Hexner, Rocks, Liu, *PRX* 11, 021045 (2021) — coupled learning (resistor + mechanical).
- Rocks, Pashine, Bischofberger, Goodrich, Liu, Nagel, *PNAS* 114 (2017) — tuning allosteric/mechanical
  networks by bond stiffness/removal.
- Stern & Murugan, *Annu. Rev. Condens. Matter Phys.* 14 (2023) — "Learning without neurons."
- Scellier & Bengio (2017) — equilibrium propagation (the `β→0` gradient limit, energy-based).
- Reuse: `self-assembly/stage1_network.py` (parity-3 patterns, two-sided separation, SNR verdict idiom);
  A's `audit.py`, `gradient_alignment`. Context: `2026-06-18-experiment-a-findings.md` (A's PASS; the
  under-powered-FORECLOSE warning; the recommendation to assert a concrete alignment margin in B's keystone).
