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
| **B2** (2026-11-05) | same lattice, offline representability re-probe | parity-3 | **FORECLOSE-REPRESENTABILITY-CONFIRMED** — B's foreclosure is not a budget/objective artifact |
| **B0** (2026-11-06) | 2D spring network, local rule vs ceiling + matched GD | mechanical allostery: attenuate / amplify / invert | **PASS / MULTISTABILITY-FORECLOSE / PASS** — the rule learns at network scale (incl. a sign inversion); amplification drives the lattice into buckling, and so does offline GD |
| **B0-noise** (2026-11-06) | same, read noise in the two bond reads | amplify | **H5-MECHANISM-CONFIRMED** — independent-read noise forecloses (by buckling) at σ = 1e-3, common-mode never |
| **A2** (2026-11-06) | heterogeneous resistor net; Poiseuille law | β-law; A's tasks | **1 − ρ ∝ β^2.00**; Poiseuille **PASS** on both tasks |

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

**Experiment B2 (re-probe, 2026-11-05).** B's representability leg rested on a fixed-scale MSE target,
plain gradient descent and a fixed budget — the three ingredients the portfolio commons has since
identified as able to manufacture a false foreclosure (`foreclosure-can-be-a-budget-artifact`). B2
re-probes it with a scale-free objective (the Walsh **parity purity** of the readout), L-BFGS with a
budget-saturation trace, a gradient-free optimizer, and the untrained lattice's Walsh spectrum vs
deformation. Result: the parity channel grows as `d³` against a linear channel `∝ d` (crossover at
d ≈ 1.1 lattice spacings vs buckling at ≈ 0.25); optimizing rest lengths and stiffnesses raises the
parity coefficient several-hundred-fold, but every optimum is **pinned on the buckling boundary** and
38/38 solvable optima have negative separation. See `docs/superpowers/specs/2026-11-05-experiment-b2-findings.md`; the
restart evaluation and ranked roadmap are in `2026-11-05-project-status-and-roadmap.md`.

**Experiment B0 (2026-11-06).** The positive control B never ran: on *mechanical allostery* the local rule
learns at network scale, 100 % physics-local — attenuation to ~1e-3 normalized error and a **sign inversion**
of the allosteric response (which needs pre-stress / geometric nonlinearity) to ~5e-3, beating matched-budget
offline GD by 30–100× on inversion. Amplification is learned to near the ceiling and then **the lattice
buckles** (2 of 3 seeds, at epochs 54 and 4434) — and the offline GD baseline buckles at nearly the same
epochs: amplifying a response means driving a stiffness toward zero, so loss descent on this medium marches
toward the stability boundary. A read-noise sweep on the same task confirms the commons' H5 two-read
fragility mechanism on an elastic substrate, with the tax paid as buckling rather than misdirection. See
`docs/superpowers/specs/2026-11-06-experiment-b0-findings.md` and `2026-11-06-experiment-b0-noise-findings.md`.
The Experiment C design (`2026-11-06-experiment-c-design.md`) proposes embracing bistability; its first
blocker, a solver that finds minima through snaps, is `mechanical_solver.settle()`.

**Why the keystones hold (commons lesson `gradient-flow-is-the-ep-license`).** Both substrates are genuine
gradient flows — the resistor net minimizes a co-content, the spring net its elastic energy — which is
what licenses the equilibrium-propagation guarantee that the local contrast *is* the gradient as β→0.
A driven / non-variational steady state would void it even at 100 % physics-local.

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
| `experiment_a2.py` | β-law on a heterogeneous net (1 − ρ ∝ β²) + the Hagen–Poiseuille law run as a headline |

**Experiment B (elastic):**

| File | Responsibility |
|---|---|
| `spring_network.py` | 2D spring energy/force/tangent-stiffness (smoothed tension–compression asymmetry) + triangulated builder |
| `mechanical_solver.py` | Damped-Newton energy minimization + line-search guard + basin-jump detection; `settle()` (LM-damped, finds minima through snaps — Experiment C entry point) |
| `mechanical_learning.py` | The two-phase mechanical rule (learns stiffness `k` and rest length `ℓ`) |
| `experiment_b.py` | Crash-tolerant driver: train, baseline, five-way verdict + JSON (alignment certified at init; reports η/β) |
| `parity_analysis.py` | Walsh decomposition of the 3-input readout: `parity_purity` (scale-free parity content) + its gradient |
| `experiment_b2.py` | Offline representability re-probe: purity / MSE under L-BFGS-B, DE, saturation trace, `verdict_b2` |
| `experiment_b0.py` | Mechanical allostery: local rule vs converged ceiling (`mech_ceiling`) and matched-budget GD; trajectory diagnostics; d-map; `verdict_b0` |
| `experiment_b0_noise.py` | Read noise in the two bond reads (independent vs common-mode) × β; `verdict_b0_noise` (H5 mechanism) |

**Shared:**

| File | Responsibility |
|---|---|
| `tasks.py` | A: allostery/regression + verdict. B: parity-3 task, two-sided separation, `verdict_b`, `verdict_b2` |
| `baselines.py` | Offline adjoint-gradient baselines (linear A; implicit-function-theorem B via `mech_readout_adjoint`, any readout loss) + gradient-alignment |
| `audit.py` | The "who computes the update?" provenance hook, projected onto the commons' two axes (gradient source × write mechanism) |
| `docs/superpowers/` | Design specs, implementation plans, and findings |

## Running

```bash
pip install -r requirements.txt
python -m pytest -q          # the full test suite (one module per source module)
python experiment_a.py       # Experiment A: prints a table + writes experiment_a_results.json
python experiment_b.py       # Experiment B: prints a per-seed table + writes experiment_b_results.json (~1 min)
python experiment_b2.py      # Experiment B2: offline representability re-probe + Walsh spectrum (~28 min)
python experiment_b0.py      # Experiment B0: mechanical allostery, local rule vs ceiling + matched GD, d-map (~26 min)
python experiment_b0_noise.py  # read-noise sweep on B0's amplify task (~14 min)
python experiment_a2.py      # Experiment A2: beta-law + Poiseuille headline (~1 s)
```

**Stack:** Python, numpy, scipy (L-BFGS-B / differential evolution in B2), pytest. The true gradients
are hand-written adjoints (no deep-learning framework), each cross-checked against finite differences.
The spring kernels are vectorized over bonds (Experiment B runs in ~1 min).

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
- Portfolio commons: `bshepp/physical-learning-commons` (`BULLETIN.md`, `synthesis/lessons-ledger.md`) —
  the lessons B2 and the two-axis audit adopt.
