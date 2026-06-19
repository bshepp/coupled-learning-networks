# Coupled-Learning Networks

Decentralized, **physics-driven** learning in passive physical networks. The question this project
answers: *when a network adapts to a task, **who computes the weight update?*** If a CPU optimizer does
it, the substrate is a peripheral. The goal is a rule where every weight change is derivable from
quantities **physically local** to each element — the substrate's own physics performs the update.

This is project `01` (the flagship) of the **physical-learning-substrates** portfolio, branching from
the co-located ionic computing line. It contains two experiments in two media.

## Status

| Experiment | Substrate | Task | Verdict |
|---|---|---|---|
| **A** | resistor / flow network (scalar potential) | allostery + regression | **PASS** — the physics computes the update |
| **B** | 2D nonlinear spring network (signed displacements) | parity-3 | **FORECLOSE** — elastic state can't clear the wall |

**Experiment A (PASS).** A passive resistor network, trained only by the local two-phase
coupled-learning rule, reaches the offline optimizer's error on allostery and regression, and its
per-step update equals the true loss gradient as the nudge `β→0` (sign-match 1.0, ρ 1.0). The audit log
confirms 100% physics-local updates. *The dual-role learning thesis is real in a substrate.* (Positive
conductances obey the maximum principle, so A is deliberately **not** for parity.)

**Experiment B (FORECLOSE).** The signed/non-monotone elastic substrate does **not** clear the parity-3
wall, in a two-part way: (1) in the stable low-deformation regime the offline optimizer cannot represent
parity-3 anywhere swept (deformation `d`, asymmetry `α`, network size, random inits → separation always
negative); (2) the deformation needed for strong non-monotonicity is exactly where the network buckles
(indefinite tangent stiffness → multistability), so the two-phase rule becomes ill-defined. The
*mechanism* is sound — the keystone confirms the local rule equals the true gradient (ρ 1.0). The limit is
the substrate, not co-located learning. → **The wall is deeper than passivity**; escalate to project `03`
(energy-based/Ising). See `docs/superpowers/specs/2026-06-19-experiment-b-findings.md`.

## The shared harness

Both experiments share the same loop and audit: *clamp inputs (free phase) → relax to equilibrium →
clamp inputs+outputs (nudged phase) → relax → update each weight from the LOCAL difference between the
two relaxed states → score.* The constitutive law is pluggable; the learning rule and the verdict are
shared. (The canonical extraction of this as a standalone `00-shared-harness` is deferred to the
portfolio-wide reconciliation.)

## Structure

**Experiment A (resistor):**

| File | Responsibility |
|---|---|
| `solver.py` | Batched Newton/Laplacian equilibrium solve + the `relax` seam |
| `network.py` | Constitutive laws (resistor, Hagen–Poiseuille), `Network`, grid builder |
| `coupled_learning.py` | The two-phase coupled-learning rule (scalar/voltage) |
| `experiment_a.py` | Driver: train, compare to baseline, emit verdict + JSON |

**Experiment B (elastic):**

| File | Responsibility |
|---|---|
| `spring_network.py` | 2D spring energy/force/tangent-stiffness (smoothed tension–compression asymmetry) + triangulated builder |
| `mechanical_solver.py` | Damped-Newton energy minimization + line-search guard + basin-jump detection |
| `mechanical_learning.py` | The two-phase mechanical rule (learns stiffness `k` and rest length `ℓ`) |
| `experiment_b.py` | Crash-tolerant driver: train, baseline, five-way verdict + JSON |

**Shared:**

| File | Responsibility |
|---|---|
| `tasks.py` | A: allostery/regression + verdict. B: parity-3 task, two-sided separation, `verdict_b` |
| `baselines.py` | Offline adjoint-gradient baselines (linear A; implicit-function-theorem B) + gradient-alignment |
| `audit.py` | The "who computes the update?" provenance hook |
| `docs/superpowers/` | Design specs, implementation plans, and findings |

## Running

```bash
pip install -r requirements.txt
python -m pytest -q          # the full test suite (one module per source module)
python experiment_a.py       # Experiment A: prints a table + writes experiment_a_results.json
python experiment_b.py       # Experiment B: prints a per-seed table + writes experiment_b_results.json
```

**Stack:** Python, numpy, pytest (scipy is a declared dependency, currently unused). The true gradients
are hand-written adjoints (no deep-learning framework), each cross-checked against finite differences.

## Method discipline

Compute/simulation first; foreclosure is a valid, cheap outcome; observation is kept separate from
interpretation; the offline-optimizer baseline is reported every time; pass/fail criteria are
pre-registered; the smoke test asserts a *valid* verdict, never a specific one. Per experiment, the
`docs/superpowers/specs/` folder holds the design spec and findings; `docs/superpowers/plans/` holds the
implementation plan.

## References

- Stern, Hexner, Rocks, Liu — *PRX* 11, 021045 (2021), "Supervised learning in physical networks."
- Dillavou, Stern, Liu, Durian — *Phys. Rev. Applied* 18, 014040 (2022), "Demonstration of
  decentralized, physics-driven learning."
- Rocks, Pashine, Bischofberger, Goodrich, Liu, Nagel — *PNAS* 114 (2017), allosteric/mechanical network
  tuning (background for Experiment B).
- Stern & Murugan — *Annu. Rev. Condens. Matter Phys.* 14 (2023), "Learning without neurons."
- Scellier & Bengio (2017), "Equilibrium propagation."
