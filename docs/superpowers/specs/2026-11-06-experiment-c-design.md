# Design — Experiment C: learning on an elastic medium that embraces bistability

**Date:** 2026-11-06
**Status:** DESIGN DRAFT (brainstorm → spec; not yet planned or implemented). Written as the natural
next question after Experiments B, B2 and B0; to be reviewed before `writing-plans`.
**Project:** `coupled-learning-networks` (portfolio `01`). Builds on B/B2/B0 and on the commons' H4
(directed aging needs an in-fabric free-phase reference) and `expressivity-multistability-pincer`.

---

## 1. Why this experiment

Three results now pin the elastic medium's problem to one place:

- **B** (2026-06-19): parity-3 forecloses in a *pincer* — the stable regime cannot represent it; the
  expressive regime is multistable and the two-phase rule becomes ill-defined.
- **B2** (2026-11-05): the parity channel of the lattice scales as `d³` against a linear `d¹`; optimizing
  stiffness and rest length grows the parity coefficient ~600× but **every optimum is pinned on the
  buckling boundary** — the parity-increasing direction in parameter space *is* the buckling direction.
- **B0** (2026-11-06, pilot): even on a *friendly* allostery task the gradient flow (local rule and GD
  alike) descends for thousands of epochs and then **walks the lattice into buckling** — learning on
  this medium tends toward the instability on its own.

So the obstacle is not that the substrate lacks non-monotonicity; it is that its non-monotonicity *is*
an instability, and the two-phase rule (free and nudged relaxations in a shared basin) cannot use it.
Two routes follow, and they are the two arms of Experiment C:

- **Arm 1 — bistable by design.** Give the lattice elements that are *intentionally* bistable (snap-through
  units with a known double-well energy) and ask whether a two-phase rule exists that is well-defined
  *across* basin changes. If non-monotonicity must come with multistability on a passive elastic medium,
  the learning rule has to be robust to it rather than avoid it.
- **Arm 2 — fast/slow two-timescale medium (commons H4).** `07`'s Phase 0b found that a directed-aging
  rule anti-aligns because a temporal baseline is *not* the free-phase reference; the diagnosed fix is a
  medium with a fast mode that physically re-plays the free relaxation each step and a slow mode that
  carries the weight. This arm tests that mechanism on the spring lattice. It is the hardware-facing arm:
  it is what a printed compliant medium would need to do.

Both arms keep the audit question central: *who computes the update?* must stay the physics, and
`is_physical` (both axes) must hold on every headline run.

## 2. Arm 1 — bistable elements and a basin-robust rule

**Substrate.** Replace a subset of bonds (or add a diagonal "snap" bond per cell) with a **double-well
element**: energy `E(e) = ¼ a e⁴ − ½ b e²` (two minima at `e = ±√(b/a)`, barrier `b²/4a`), or the
geometric version — a shallow von-Mises truss (two bars meeting at a small angle) whose apex snaps
between two stable positions. Trainable: `k`, `ℓ` on ordinary bonds; `b` (barrier height / well depth)
and the asymmetry of the wells on snap elements. Everything remains `C∞` so the implicit adjoint stays
valid *within* a basin.

**Why it could clear the wall.** A snap element's apex position is a signed, strongly non-monotone
function of the load — exactly the response the parity channel needs — without requiring the *whole*
lattice to be near buckling. B2's d³ law is a property of smooth small-deformation elasticity; a
designed bistable element breaks it locally, at a controllable location.

**The rule problem (the real bet).** With bistability the free and nudged phases can land in different
basins. Candidate rules, to be compared against the offline ceiling *and* a positive control (the same
rule on a monostable lattice must recover ρ→1):

1. *Warm-started two-phase, basin-gated* (B's rule): update only when `basin_jump` is false; count jumps.
   Baseline — expected to degrade as bistability is engaged.
2. *Symmetric nudge* (`±β`, EP's centered difference): the contrast `E(+β) − E(−β)` is odd in β; if both
   nudges stay in the free basin the gradient estimate is second-order accurate, and a jump in one
   direction is detected as asymmetry. Cheap, and 04 uses the quadrature form already.
3. *Basin-averaged contrast*: when the nudged phase jumps, relax from *several* warm starts (the free
   state plus small perturbations) and take the lowest-energy nudged state — a physical "settling" model.
   Costs relaxations, but every quantity is still local.
4. *Hysteretic (snap-counting) rule*: treat a snap event as the signal — update a snap element's
   asymmetry by the sign of the snap it made in the nudged phase relative to the free phase. This is a
   genuinely new local rule and the most speculative.

**Pre-registered verdicts (draft).**

| Verdict | Condition |
|---|---|
| PASS | parity-3 two-sided separation SNR ≥ 3 across seeds by a rule from the list, with `is_physical` on both axes, init alignment ρ ≥ 0.8 on the positive control, and the learned configuration *robust* (separation survives ±10 % parameter noise) |
| FORECLOSE-RULE | the offline ceiling separates parity-3 (so the bistable substrate *represents* it) but no candidate rule learns it → multistability defeats local two-phase learning even when designed in |
| FORECLOSE-REPRESENTABILITY | the ceiling cannot separate parity-3 either (bistable elements are not enough) |
| AMBIGUOUS | marginal / single-seed / rule works only with a hand-set snap schedule |

The FORECLOSE-RULE outcome is the scientifically sharp one: it would say that on passive elastic media
the parity wall is a property of *two-phase learning*, not of representability.

## 3. Arm 2 — fast/slow two-timescale medium (H4)

**Model.** Each bond carries a *fast* internal variable `f` (relaxes on timescale τ_f ≪ step) and a *slow*
weight `w` (τ_s ≫ step). The lattice is driven with inputs clamped; the fast variable tracks the current
equilibrium (`f → e_free` each step — this *is* the free-phase reference, held in the fabric); the output
is then nudged and the slow weight ages by the local difference between the instantaneous extension and
the fast variable's stored value: `ẇ ∝ −(e_now − f)·∂e/∂w`-type law (to be derived from a chosen aging
energy so it is a gradient flow — the EP license). Compare against (a) a plain temporal-EMA aging rule
(07's anti-aligned case, as the negative control) and (b) B's explicit two-phase rule (positive control).

**Pre-registered verdicts (draft).** PASS if the two-timescale rule's update aligns with the true
gradient (ρ ≥ 0.8 at init, sign ≥ 0.9) *and* learns allostery to the matched-budget baseline with
`is_physical` on both axes; FORECLOSE if alignment is at chance or anti-aligned while the explicit
two-phase control aligns; AMBIGUOUS otherwise. A read-noise sweep (as in `experiment_b0_noise.py`) is
part of the headline, since the whole point of the single-read medium is noise robustness.

## 4. Shared harness changes needed

- `spring_network`: a per-bond constitutive *family* (ordinary, double-well, von-Mises) with
  energy/force/Hessian each FD-gated; `relax_mech` unchanged (it is energy-agnostic).
- `mechanical_solver`: a `settle()` helper (multi-start relaxation returning the lowest-energy state)
  for rule 3; a basin label (sign pattern of snap-element states) for diagnostics.
- `mechanical_learning`: rule variants behind one signature; the audit tags every variant
  `PHYSICS_LOCAL` only if its inputs are bond-local (rule 3's settling is local per relaxation; rule 4's
  snap sign is local).
- `tasks`: parity-3 on the bistable lattice reuses `parity3_mech_task`'s I/O; allostery for Arm 2 reuses
  `allostery_mech_task`.
- Verdict logic as drafted above, in `tasks.py`, fixed before any headline run.

## 5. Risks and what would kill it early

- Double-well bonds with the Newton solver: indefinite Hessians *are* the design here, so `relax_mech`'s
  `RuntimeError` on a failed line search will fire constantly; the solver needs a damped/trust-region
  path that *finds a minimum* rather than raising — while keeping B's crash-tolerance semantics for the
  ordinary lattice. This is the first thing to prototype.
- The hysteretic rule (4) may not be a gradient of anything; the commons' `gradient-flow-is-the-ep-license`
  says then to *measure* its alignment and report it as a heuristic, never assert it.
- Arm 2's aging law must be derived from an energy so its alignment is licensed; a hand-written rule that
  "looks local" but anti-aligns is 07's trap.

## 6. Cheapest first step (one session)

Prototype a single von-Mises snap cell (3 nodes) inside `spring_network`, FD-gate its energy/force/Hessian,
show `relax_mech` (or a damped variant) finds both minima from different starts, and measure rule 2's
(symmetric nudge) alignment on a 3×3 lattice with one snap cell against the implicit adjoint — in-basin
and across a snap. That single number (ρ across a snap) decides whether Arm 1 is worth a full plan.

## 7. References

- Stern, Hexner, Rocks, Liu, *PRX* 11, 021045 (2021); Scellier & Bengio (2017) — two-phase / EP.
- Pashine, Hexner, Liu, Nagel, "Directed aging, memory, and nature's greed," *Sci. Adv.* (2019) — aging as
  learning in materials (Arm 2).
- Howell, *Compliant Mechanisms* (2001); bistable/von-Mises trusses — designed bistability (Arm 1).
- This repo: `2026-06-19-experiment-b-findings.md`, `2026-11-05-experiment-b2-findings.md`,
  `2026-11-06-experiment-b0-findings.md` (when written); commons `synthesis/hypotheses.md` H4,
  `lessons-ledger.md` (`expressivity-multistability-pincer`, `gradient-flow-is-the-ep-license`,
  `network-foreclose-needs-a-positive-control`).
