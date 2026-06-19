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
python -m pytest -q                                   # full suite (~44 tests, one module per source module)
python -m pytest tests/test_coupled_learning.py -q    # one module
python -m pytest tests/test_coupled_learning.py::test_training_reduces_error   # one test
python experiment_a.py        # Experiment A driver: prints table, writes experiment_a_results.json
python experiment_b.py        # Experiment B driver: prints per-seed table, writes experiment_b_results.json
```

Stack: Python + numpy + pytest (scipy is declared but currently unused). No deep-learning framework — all
"true" gradients are hand-written adjoints. `*_results.json` are gitignored scratch outputs.

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

Shared: `tasks.py` (task definitions, error/separation metrics, **`verdict()` and `verdict_b()`**) and
`audit.py` (the provenance hook). `tasks.py` and `baselines.py` each hold both A and B sections in one file.

### The audit invariant — do not break it

`audit.py` tags every weight change as `PHYSICS_LOCAL` or `OFFLINE_GRADIENT`. The headline training functions
(`train`, `train_mech`) record `PHYSICS_LOCAL`; only the baseline controls (`gd_baseline`, `gd_baseline_mech`)
record `OFFLINE_GRADIENT`. **The smoke tests assert `audit_fraction_physics_local == 1.0`** — if you route any
offline gradient into the headline path, those tests fail *by design*, because that would falsify the thesis.
Keep the baseline's audit log separate from the headline run's.

### The gradient keystone

The local rule is only meaningful if it equals the true loss gradient as `β→0`. `baselines.py` computes the
exact adjoint gradient (hand-written, **cross-checked against finite differences** in the tests — keep that
check passing when touching `forces`/`tangent_stiffness`/`mech_adjoint_gradient`). `gradient_alignment()`
reports `sign_match`/`rho`/`cosine`; the verdict requires high alignment for a PASS.

## Conventions and gotchas

- **Flat module layout.** Source modules live at repo root and import each other by bare name (`from solver
  import relax`). `conftest.py` inserts the repo root on `sys.path` so this works under any pytest import mode —
  don't convert to a package or add `src/` without updating that.
- **One test module per source module.** New source module ⇒ new `tests/test_<module>.py`.
- **Pre-registration discipline.** Pass/fail thresholds live in `verdict()`/`verdict_b()` and are fixed ahead of
  the run. Smoke tests assert a *valid* verdict (one of the allowed statuses), **never a specific outcome** —
  the lone exception is `test_experiment_a.py` asserting regression PASSes (a friendly, representable task).
  Don't "fix" a test by making it assert the verdict you happened to get.
- **Crash-tolerance is intentional in Experiment B.** Solver `RuntimeError` (singular/indefinite tangent
  stiffness, failed line search) is a real physical event — buckling/multistability — and is *counted* as an
  instability (jump rate), not masked. Catches are deliberately narrowed to `RuntimeError` so genuine bugs
  still surface; keep them narrow.
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

**At session start, read:**
- `../00-shared-harness/BULLETIN.md` — what's newly available + the **Recommended adoptions** (find this
  project's entry, incl. the `Hardware — 01` notes: the FPAA + compliant/directed-aging flow leads).
- `../00-shared-harness/synthesis/glossary.md` — shared vocabulary; use these terms.

**Check when relevant:** `../00-shared-harness/synthesis/verdict-registry.md` (portfolio scoreboard),
`lessons-ledger.md` (e.g. `audit-is-two-axis`, `nudge-strength-is-task-scaled`,
`idealized-pass-is-not-realistic-pass`), and `hardware-paths.md`.

**Report back:** when you produce a reusable finding or module, leave it discoverable (a dated
findings/DEVLOG doc) so the commons can harvest it.
