# SEED — 01 · Coupled-Learning Networks (flow / resistor + elastic / spring)

**Status:** SEED (turn into a spec with `brainstorming` first).
**Depends on:** `00-shared-harness`.
**Priority:** FLAGSHIP — do this first. Nearly free (reuses the solver) and it tests the *thesis itself*.

## Goal

Demonstrate that a decentralized, **local, physics-driven** learning rule — no global optimizer — adapts
a passive network to a task, in two media:

- **Experiment A — flow / resistor network (the mechanism demo).** A scalar-potential network (resistors
  or Hagen–Poiseuille flow) trained by coupled learning on a **friendly task** (regression / allosteric
  source→target response). *Caveat:* positive conductances share the parity maximum-principle wall, so
  **A is not for parity** — it validates that the physics computes the update on a task the substrate can
  represent.
- **Experiment B — elastic / spring network (the signed escape).** Node *displacements* are vector and
  signed; nonlinearity (geometric / buckling / tension-compression) can make the response **non-monotone**.
  Attempt **parity-3** here — this is the variant that might clear the wall the ionic line hit.

## The question / verdict

- A: does the local two-phase rule drive task error to the offline-optimizer baseline? → *the dual-role
  learning thesis is real in a substrate.* (The cleanest, cheapest demonstration of "who computes the
  update.")
- B: does signed/non-monotone elastic state achieve a feasible parity-3 separation that passive conductance
  networks provably cannot? → *the wall is about passivity, not co-located learning.*

## Physics / simulation

- Flow/resistor: `I = ΔV / R` (resistor) or `Q = ΔP · (πr⁴/8μL)` (Poiseuille) — both the **same Laplacian
  solve** as `self-assembly/stage2_solver.py`. Coupled-learning update on edge conductances/radii.
- Elastic: node force balance for a spring network, `F_ij = k_ij·(|x_i−x_j| − ℓ_ij)` with optional
  geometric nonlinearity; solve for equilibrium displacements (Newton). Learn `k_ij` and/or rest lengths
  `ℓ_ij` by the contrastive rule.
- Use `00-shared-harness` for the learning loop, task harness, and the audit hook.

## Pre-registered verdicts

- **PASS (A):** local rule ≈ offline baseline on regression/allostery.
- **PASS (B):** feasible, robust positive parity-3 separation in the elastic network.
- **FORECLOSE (B):** even signed elastic state can't separate parity-3 within physical limits → the wall
  is deeper than passivity (escalate to active/energy-based, project `03`).

## Reuse / references

- Reuse: `00-shared-harness`, `self-assembly/stage2_solver.py`, parity harness.
- Stern et al. *PRX* 2021; Dillavou et al. *PRApplied* 2022; Stern & Murugan, *Annu. Rev. Cond. Mat.*
  2023 "Learning without neurons" (review); allosteric/mechanical learning (Rocks, Liu, Nagel).

See `../README.md` for shared conventions.
