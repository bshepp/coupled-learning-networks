# Design — 01 · Coupled-Learning Networks (Experiment A + shared harness)

**Date:** 2026-06-18
**Status:** DESIGN (approved; next step `writing-plans`).
**Source seed:** `../../../SEED.md` (project 01). Portfolio conventions: `../../../../README.md`.
**Depends on:** `00-shared-harness` — **folded into this project** (built here, extracted later only if `02`/`03` need it).

---

## 1. Scope & intent

This spec covers **Experiment A (the mechanism demo) plus the shared learning harness**. It does **not**
cover Experiment B (the elastic/spring parity-3 escape); B gets its own spec once the harness is validated
by A. Rationale: A and B have very different cost/risk profiles, and B requires a genuinely new vector
nonlinear solver — bundling them would couple a near-foregone result to a speculative one.

**The thesis under test (portfolio throughline):** a learning substrate where *the signal being processed
is the same physical event that reshapes the processor* — compute and weight-update co-located. **The
audit question, asked of every weight change: *who computes the update?*** If a CPU optimizer produces the
weight change, the substrate is a peripheral, not the learner.

**What A demonstrates:** a decentralized, **local, physics-driven** coupled-learning rule (no global
optimizer) adapts a passive resistor/flow network to a *friendly* task, and the local update is a faithful
image of the true loss gradient. This is the cleanest, cheapest demonstration of "the physics computes the
update."

**Honest framing of A's stakes.** A's PASS is *near-foregone* — Dillavou et al. (2022) already demonstrated
decentralized physics-driven learning in resistor networks. A's primary value here is therefore **(a)
validating the shared harness and the audit hook** that B and later projects depend on, and **(b)** a clean,
self-contained reproduction with an explicit gradient-alignment measurement. The genuine scientific bet of
project 01 lives in B (deferred).

---

## 2. Architecture (Approach 3 — hybrid reuse)

Reuse the proven batched Newton/Laplacian equilibrium solve from `self-assembly/stage2_solver.py`, behind
**one thin `relax` seam**, and write the genuinely new, load-bearing piece — the **coupled-learning rule** —
once, with a substrate-agnostic signature. Do **not** build a speculative substrate abstraction for B before
B has told us what it needs; leave exactly one clean seam (`relax`) where the vector spring solver plugs in
later.

### Module layout

House code pattern: small focused modules; functions return numbers; each runnable module has a `main()`
that prints a table + dumps JSON; one pytest module per source module. All files in
`01-coupled-learning-networks/`.

| Module | Purpose | Origin |
|---|---|---|
| `network.py` | Substrate description: nodes, edges, weights, clamp/boundary spec; builders for small random/lattice nets with designated input/output nodes; the **pluggable constitutive law** (resistor `I=gΔV`; Hagen–Poiseuille as a thin second law) | new |
| `solver.py` | The **`relax(substrate, clamps) → node_state`** seam | **port** of `stage2_solver.py` (batched Newton/Laplacian solve) |
| `coupled_learning.py` | **The heart**: `free_phase`, `nudged_phase`, `local_update` — the two-phase contrastive rule on edge-local power | new |
| `tasks.py` | Allostery + regression task defs (node assignments, datasets, error metrics) | new; metric idioms from `stage1_network.py` |
| `baselines.py` | Offline true-gradient (adjoint) + gradient-descent baseline; gradient-alignment statistics | new |
| `audit.py` | The audit hook — tags every weight update `physics-local` vs `offline-gradient` | new |
| `experiment_a.py` | Driver: trains on both tasks, compares to baseline, emits verdict table + JSON | new |

### Reuse boundary: port, don't import

The portfolio is a **separate git repository** from `self-assembly`. Rather than a fragile cross-repo
import, **port (copy + adapt, with attribution in the module docstring)** the relevant ~50 lines of
`stage2_solver.py` into `solver.py`, and the `separation`/`verdict` idioms of `stage1_network.py` into
`tasks.py`/the verdict logic. This keeps `01` self-contained and independently `git init`-able (a portfolio
convention).

### The `relax` seam

`relax(substrate, clamps) → node_state` wraps the batched Newton solve. For a **linear** resistor network
(`φ≡1`) the Newton solve degenerates to a single linear Laplacian solve (the solver's warm-start step is the
answer). Batched over all task patterns at once. This is the single seam where Experiment B's vector
nonlinear spring solver will later plug in, unchanged in signature.

---

## 3. The coupled-learning rule (the heart)

The one module that makes the thesis honest. Three steps per training example (Stern, Hexner, Rocks, Liu,
*PRX* 2021 form):

1. **Free phase** — clamp input nodes to the input pattern; output nodes float. `relax` → free state `V^F`;
   read free output `V^F_out`.
2. **Nudged phase** — keep inputs clamped; nudge outputs a *small* step toward the target by clamping them
   to `V^N_out = V^F_out + β·(target − V^F_out)`, small nudge `β`. `relax` → nudged state `V^N`.
3. **Local update** — for every edge, using *only* the voltage drop across that edge in the two phases:

   ```
   Δg_ij = (η / β) · [ (ΔV^F_ij)² − (ΔV^N_ij)² ]
   ```

   `η` = learning rate (separate from nudge `β`). The per-edge quantity is edge power — physically available
   at the element. **No global error signal touches the weight.** This is the audit point made concrete.

Two correctness commitments:

- **Sign is verified, not asserted.** The `±` convention is pinned by checking the update descends the loss
  against the adjoint gradient (§5). A sign error fails the gradient-alignment test — the test *is* the
  proof.
- **`β`-sweep.** Coupled learning equals the true gradient only as `β→0` (the equilibrium-propagation limit;
  Scellier & Bengio 2017). The spec runs a small `β` sweep to (a) show the alignment-vs-`β` tradeoff
  (smaller `β` → better alignment, more numerical noise) and (b) pick the operating `β`.

---

## 4. Tasks & data flow

Two friendly tasks, both representable by a passive positive-conductance **linear** network:

- **Allostery** (literature-canonical): designate source node(s) and *distant* target node(s); train
  conductances so a source drive produces a specified target response (e.g. a desired source→target ratio
  `r`) across a set of source amplitudes. The Stern/Dillavou demonstration task.
- **Regression**: map a few input-node voltages to an output-node voltage over a dataset drawn from a
  **linear** ground-truth map (friendly = representable by the linear substrate).

### Data flow (per run)

```
build_network (topology, I/O nodes)  →  init weights g>0
   └─ for epoch in epochs:
        free_phase(inputs)         → V^F, output^F
        nudged_phase(inputs, tgt)  → V^N
        Δg = local_update(V^F, V^N) ;  g ← clip(g + Δg, g_min, g_max)   # g>0 enforced
        record task_error(epoch)
   →  evaluate vs baseline  →  verdict + JSON
```

Full-batch is fine — `solver` is batched over patterns.

**Passivity made explicit.** Conductances are clipped to `[g_min, g_max]` with `g_min > 0`. This is the
physical reason A cannot do parity (the maximum-principle wall) — stated as a deliberate boundary, not an
accident, and the reason parity work is reserved for B.

**Error metric.** Normalized MSE between output and target (relative error for allostery). The
`separation`/SNR `verdict` idiom from `stage1_network.py` is reserved for B's parity work; A uses
error-vs-baseline, not class separation.

---

## 5. Baselines & verdict logic

**True gradient (offline reference).** The task loss `L = ½·Σ(V_out − target)²` has an exact, cheap gradient
`∂L/∂g_ij` via the **adjoint method** through the linear Kirchhoff solve (resistor networks are reciprocal,
so the adjoint reuses the forward factorization). A **finite-difference** gradient on a tiny network is the
test-time cross-check that the adjoint is correct.

**Offline optimizer baseline = the "CPU computes the update" control.** Gradient descent on the adjoint
gradient, trained to convergence → the best achievable task error (the representability + learnability
ceiling). Explicitly tagged `offline-gradient` by the audit hook (allowed only as the baseline).

**Gradient alignment (mechanism check, mirroring Phase 2a).** At matched weights, compare the local update
`Δg_local` to `−∂L/∂g`:
- **sign-match fraction** (edges where signs agree),
- **Pearson ρ / cosine similarity**,

measured across the `β`-sweep (expect → 1.0 as `β→0`).

### Pre-registered verdict

| Verdict | Condition |
|---|---|
| **PASS** | local-rule final error ≤ baseline × (1 + `tol`) **AND** sign-match ≥ 0.95, ρ ≥ 0.9 at operating `β` → *the physics computes the update* |
| **FORECLOSE** | local rule cannot materially reduce error where the baseline can → *representable but not locally learnable* (a distinct, valuable result — separates representability from learnability) |
| **AMBIGUOUS** | learns but misses the baseline, or alignment is weak |

`tol` (~10–20 % relative) and the alignment thresholds are config constants. A linear substrate should give
near-perfect alignment, so the bar is set high deliberately.

---

## 6. The audit hook

The honesty mechanism that makes "who computes the update?" a *logged, checkable fact*. A lightweight
recorder each run holds; every weight change is tagged at its source:

- `physics-local` — from `coupled_learning.local_update` (edge-local two-phase quantities only),
- `offline-gradient` — from the adjoint/CPU baseline (allowed *only* in the baseline run).

The run's JSON summary reports the provenance breakdown: the **headline coupled-learning run must be 100 %
`physics-local`** (asserted at run end), the baseline run is `offline-gradient`, and any mixed run is flagged
as a bug. The thesis claim becomes mechanically verifiable, not asserted in prose.

---

## 7. Testing strategy

TDD; one test module per source module; numpy + scipy + pytest (matching `self-assembly`). **No new heavy
dependencies** — the adjoint gradient is hand-written (no jax).

| Test module | Key checks |
|---|---|
| `test_solver.py` | `relax` matches a hand-solved small network; linear → single solve; convergence; reciprocity |
| `test_network.py` | builders/topology/I-O assignment valid; resistor & Poiseuille laws correct |
| `test_coupled_learning.py` | **keystone** — `Δg_local` → `−adjoint gradient` as `β→0` (sign-match→1, ρ→1); free/nudged relax; update reduces loss |
| `test_tasks.py` | allostery + regression datasets & error metrics; untrained-network null sanity |
| `test_baselines.py` | adjoint == finite-difference; GD baseline converges on a representable task |
| `test_audit.py` | provenance tagging; mixed-run flagged; headline run 100 % `physics-local` |
| `test_experiment_a.py` | end-to-end smoke on a tiny config → verdict + JSON |

The `test_coupled_learning.py` keystone (local rule ≡ true gradient as `β→0`) carries the whole thesis.

---

## 8. Repository setup (after the implementation plan)

Once the implementation plan exists, create a proper repository for this project:

- **Name:** `coupled-learning-networks` (drops the `01-` portfolio-ordering prefix, per user request).
- **Contents:** full repo skeleton — source modules, `tests/`, `docs/`, `README.md`, `requirements.txt`,
  and a Python **`.gitignore`** (`__pycache__/`, `*.pyc`, `.pytest_cache/`, venv, results scratch, etc.).
- **GitHub:** description + topics set on the remote.
- **Outward-facing confirmation gate.** Creating a GitHub remote is a publish action. Before pushing,
  confirm **name, visibility (public/private), and topics** with the user. Do not push without that
  confirmation.

---

## 9. Out of scope / deferred

- **Experiment B** (elastic/spring, signed/non-monotone state, parity-3) — its own spec after A validates
  the harness. Known B-specific risks to carry into that spec: (1) the `relax` seam needs a **new vector
  nonlinear-spring solver** (geometric stiffness Jacobian) — *not* a constitutive-law swap on the scalar
  Laplacian; (2) **multistability** — a buckling network has multiple equilibria, so free and nudged phases
  can land in different basins, making the contrastive difference ill-defined. B's pre-registered verdicts
  must distinguish "multistability breaks the two-phase rule" from "signed state still can't separate
  parity-3."
- **Nonlinear conductance** (`g = w·φ(ΔV)`) — the ported solver supports it, but it is out of scope for A's
  linear mechanism demo and verdict.
- **Hardware** — compute/simulation first, per portfolio convention.

---

## 10. References

- Stern, Hexner, Rocks, Liu, *PRX* 11, 021045 (2021) — "Supervised learning in physical networks" (coupled
  learning rule).
- Dillavou, Stern, Liu, Durian, *Phys. Rev. Applied* 18, 014040 (2022) — "Demonstration of decentralized,
  physics-driven learning" (resistor-network hardware; allostery + regression).
- Stern & Murugan, *Annu. Rev. Condens. Matter Phys.* 14 (2023) — "Learning without neurons" (review).
- Scellier & Bengio (2017) — "Equilibrium propagation" (the `β→0` gradient limit).
- Rocks, Liu, Nagel — allosteric/mechanical network tuning (background for Experiment B).
- Reuse: `self-assembly/stage2_solver.py` (batched Newton/Laplacian), `stage1_network.py` (parity harness,
  two-sided separation, verdict). Foreclosure context: `self-assembly/docs/superpowers/specs/2026-06-18-phase2b-stage2a-findings.md`.
