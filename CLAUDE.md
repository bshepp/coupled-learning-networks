# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A scientific compute experiment, not a product. The thesis under test: a passive physical network can be
trained by a **local, physics-driven** rule where *the substrate's own physics computes the weight update* —
no global optimizer touches a weight. Two experiments probe this in two media (A = resistor/flow, B =
elastic/spring). Outcomes are pre-registered verdicts (PASS / FORECLOSE / AMBIGUOUS); **a clean FORECLOSE is a
valid, valued result** — the goal is the truth, not a PASS. Read `README.md` and the per-experiment specs in
`docs/superpowers/specs/` before changing experiment logic.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # full suite (~70 tests, one module per source module, ~12 s)
python -m pytest tests/test_coupled_learning.py -q    # one module
python -m pytest tests/test_coupled_learning.py::test_training_reduces_error   # one test
python experiment_a.py        # Experiment A driver: prints table, writes experiment_a_results.json
python experiment_b.py        # Experiment B driver: prints per-seed table, writes experiment_b_results.json (~1 min)
python experiment_b2.py       # Experiment B2: offline representability re-probe (scipy L-BFGS-B/DE), writes experiment_b2_results.json
python experiment_b0.py       # Experiment B0: mechanical allostery (local rule vs mech_ceiling + matched GD, d-map), ~26 min
python experiment_b0_noise.py # read-noise sweep (independent vs common-mode) on B0's amplify task, ~14 min
python experiment_a2.py       # Experiment A2: beta-law on a heterogeneous net + Poiseuille headline, ~1 s
```

Stack: Python + numpy + scipy + pytest. No deep-learning framework — all "true" gradients are hand-written
adjoints. `*_results.json` are gitignored scratch outputs. Spring kernels are vectorized over bonds
(`edge_array`/`_bond_geometry` in `spring_network.py`) — keep new per-bond code vectorized; the FD gate
tests are the safety net.

## Architecture

**One shared two-phase loop, two pluggable substrates.** Every headline run is: *clamp inputs (free phase) →
relax to equilibrium → additionally clamp outputs a step `β` toward target (nudged phase) → relax → update
each weight from the LOCAL contrast between the two relaxed states → score against a pre-registered verdict.*
The constitutive law and equilibrium solver are swappable; the learning rule, audit, and verdict logic are
shared in spirit across both experiments.

Two parallel module families, distinguished by the `_mech` / `mechanical_` / `spring_` naming:

| Concern | Experiment A (scalar voltage) | Experiment B (vector displacement) |
|---|---|---|
| Equilibrium solve (the seam) | `solver.py` → `relax()` (batched Newton/Laplacian) | `mechanical_solver.py` → `relax_mech()` (damped Newton + line search) |
| Substrate / constitutive law | `network.py` (`Law`, resistor/Poiseuille, grid builder) | `spring_network.py` (energy/force/tangent-stiffness, smoothed asymmetry, triangulated lattice) |
| Two-phase local rule | `coupled_learning.py` | `mechanical_learning.py` |
| Offline-gradient control | `baselines.py` (linear adjoint) | `baselines.py` (implicit-function-theorem adjoint) |
| Driver + verdict emit | `experiment_a.py` | `experiment_b.py` |
| Representability re-probe (offline only) | — | `experiment_b2.py` + `parity_analysis.py` (Walsh parity purity, `verdict_b2`) |
| Friendly-task positive control | `experiment_a.py` (A), `experiment_a2.py` (β-law, Poiseuille) | `experiment_b0.py` (allostery; `verdict_b0`), `experiment_b0_noise.py` (H5; `verdict_b0_noise`) |
| Converged representability ceiling | — | `baselines.mech_ceiling` (L-BFGS-B + stability barrier) — use beside any plain-GD baseline |
| Minimum-finding through snaps (Experiment C) | — | `mechanical_solver.settle()` (LM-damped; `relax_mech` stays the raising Newton solver) |

Shared: `tasks.py` (task definitions, error/separation metrics, **`verdict()` and `verdict_b()`**) and
`audit.py` (the provenance hook). `tasks.py` and `baselines.py` each hold both A and B sections in one file.

### The audit invariant — do not break it

`audit.py` tags every weight change as `PHYSICS_LOCAL` or `OFFLINE_GRADIENT` (projected onto the commons'
two axes, gradient source × write mechanism; `summary()["is_physical"]` is the both-axes gate).
`experiment_b2.py` is **all** `OFFLINE_GRADIENT` by design — it is a representability probe, not a learning
run; its test asserts `all_offline`. The headline training functions
(`train`, `train_mech`) record `PHYSICS_LOCAL`; only the baseline controls (`gd_baseline`, `gd_baseline_mech`)
record `OFFLINE_GRADIENT`. **The smoke tests assert `audit_fraction_physics_local == 1.0`** — if you route any
offline gradient into the headline path, those tests fail *by design*, because that would falsify the thesis.
Keep the baseline's audit log separate from the headline run's.

### The gradient keystone

The local rule is only meaningful if it equals the true loss gradient as `β→0`. `baselines.py` computes the
exact adjoint gradient (hand-written, **cross-checked against finite differences** in the tests — keep that
check passing when touching `forces`/`tangent_stiffness`/`mech_adjoint_gradient`). `gradient_alignment()`
reports `sign_match`/`rho`/`cosine`; the verdict requires high alignment for a PASS. **Measure alignment at
init, not at convergence** (commons pattern `alignment-certification`: at a loss minimum both vectors → 0 and
ρ is meaningless — Experiment B's seed 0 read ρ 0.000 at the end and 1.000 at init).

## Conventions and gotchas

- **Flat module layout.** Source modules live at repo root and import each other by bare name (`from solver
  import relax`). `conftest.py` inserts the repo root on `sys.path` so this works under any pytest import mode —
  don't convert to a package or add `src/` without updating that.
- **One test module per source module.** New source module ⇒ new `tests/test_<module>.py`.
- **Pre-registration discipline.** Pass/fail thresholds live in `verdict()`/`verdict_b()` and are fixed ahead of
  the run. Smoke tests assert a *valid* verdict (one of the allowed statuses), **never a specific outcome** —
  the lone exception is `test_experiment_a.py` asserting regression PASSes (a friendly, representable task).
  Don't "fix" a test by making it assert the verdict you happened to get.
- **Crash-tolerance is intentional in Experiment B — but only AFTER the robust seam.** A Newton
  `RuntimeError` from `relax_mech` can be a cold-start artifact (2026-11-06: a unique, stable equilibrium
  that Newton could not reach from the reference geometry was being counted as buckling; this flipped B's
  headline from MULTISTABILITY-FORECLOSE to FORECLOSE-REPRESENTABILITY). `free_phase_mech` therefore
  warm-starts from the previous equilibrium and falls back to `settle()`; only a failure of `settle()` too is
  a genuine instability and is *counted* (jump rate), not masked. Catches stay narrowed to `RuntimeError` so
  real bugs surface. Never reintroduce a cold-start-only free phase.
- **Offline ceilings must be budget-saturated and scale-free** before a FORECLOSE is recorded off them
  (commons `foreclosure-can-be-a-budget-artifact`); B2 is the template (saturation trace + scale-free
  objective + second optimizer class).
- **Loss descent on the elastic medium marches toward buckling** (B0: amplification learned to near-ceiling,
  then the lattice buckles — local rule AND offline GD alike; larger d_max → sooner). Always record the
  trajectory (`best_error`, `first_jump_epoch`, `jump_rate_tail`), never the endpoint alone, and gate
  multistability on the TAIL jump rate so a late blow-up is labelled as instability, not learnability.
- **Compare the local rule to a matched-budget GD baseline (lr = η, same epochs) AND to the converged
  `mech_ceiling`**; plain GD at a fixed budget understated representability by 100× in B0's pilots.
- **Knife-edge solves near buckling are not bit-reproducible** across numpy builds (seed-1 baseline in B
  flips −5.9e-02 ↔ −inf with summation order). Don't chase such flips as bugs; record them.
- **Observation vs interpretation stay separate.** Per-experiment `docs/superpowers/specs/*-findings.md` files
  split "Observed" (numbers) from "Interpretation" (claims). Match that when recording results.

## Portfolio context

Project `01` (flagship) of the `physical-learning-substrates` portfolio. The shared two-phase harness is
duplicated across both experiments here on purpose; canonical extraction into a `00-shared-harness` is
deferred to a portfolio-wide reconciliation. Sibling projects (`02`–`05`) run on parallel tracks.

## The commons — read at session start

`00` (`../00-shared-harness/`, repo `physical-learning-commons`) is the portfolio's shared **commons**:
it harvests cross-project lessons and advertises what each project should adopt. It *reads* this repo but
never writes to it — adoption is yours to apply (pull model).

The commons may not be checked out beside this repo (it was not on 2026-11-05); it is public at
`https://github.com/bshepp/physical-learning-commons` — clone it read-only to `../00-shared-harness` first.

**At session start, read:**
- `../00-shared-harness/BULLETIN.md` — what's newly available + the **Recommended adoptions** (find this
  project's entry, incl. the `Hardware — 01` notes: the FPAA + compliant/directed-aging flow leads).
- `../00-shared-harness/synthesis/glossary.md` — shared vocabulary; use these terms.

**Check when relevant:** `../00-shared-harness/synthesis/verdict-registry.md` (portfolio scoreboard),
`lessons-ledger.md` (e.g. `audit-is-two-axis`, `nudge-strength-is-task-scaled`,
`idealized-pass-is-not-realistic-pass`), and `hardware-paths.md`.

**Report back:** when you produce a reusable finding or module, leave it discoverable (a dated
findings/DEVLOG doc) so the commons can harvest it.
