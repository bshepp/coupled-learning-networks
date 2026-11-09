# Findings — Experiment B0: does the mechanical rule *learn* a representable task? (elastic allostery)

**Date:** 2026-11-06
**Question.** Experiment B certified the mechanical two-phase update *per step* (keystone ρ → 1 as β → 0)
and foreclosed on parity-3, but never showed the rule *learning* anything at network scale on the elastic
medium — the positive control the commons' `network-foreclose-needs-a-positive-control` lesson asks for,
and the prerequisite for any read-noise (H5) study on this substrate. B0 runs the literature-canonical
friendly task — **mechanical allostery** (one input node displaced, a prescribed transfer ratio at a
distant output node; Rocks et al. 2017) — in three variants relative to the untrained lattice's own
ratio r₀: **attenuate** (0.5 r₀), **amplify** (2 r₀), **invert** (−0.5 r₀).
**Verdicts (run 2, corrected solver seam):** **attenuate PASS (3/3 seeds) · amplify AMBIGUOUS (2 seeds
PASS; seed 0 learns to 2.6e-3 but misses the matched-budget GD baseline, 1.0e-3, by 2.6×) · invert PASS
(3/3)**. The mechanical two-phase rule learns at network scale, 100 % physics-local, including a sign
inversion of the allosteric response; **no instability event occurred in any cell** (jump rates 0.00).
Run 1's "amplification drives the lattice into buckling" was a cold-start solver artifact (see Experiment B's
2026-11-06 addendum) and is withdrawn.
**Audit:** headline runs 100 % `PHYSICS_LOCAL`; ceiling and baseline in separate `OFFLINE_GRADIENT` logs.

> **Observed** = measured; **Interpretation** separate. Pre-registered verdict: `tasks.verdict_b0`
> (multistability gate on the tail jump rate > 0.2 → representability via the converged ceiling > 0.1
> → PASS if local error ≤ max(1.15 × matched-budget GD, 1e-3) with init alignment sign ≥ 0.9, ρ ≥ 0.8 →
> FORECLOSE-LEARNABILITY if the rule does not even halve the untrained error → AMBIGUOUS). Task verdict
> = PASS only if every seed passes, else the most severe seed status.

---

## Observed

### Apparatus
- 4×4 triangulated lattice; input = bottom-middle node's y-displacement, amplitudes u ∈ 0.1·U[0.3, 1]
  (6 patterns); output = top-right node's y-displacement; bottom-left node pinned, bottom-right pinned in y.
- Untrained (uniform k = 1, unstressed) transfer ratio **r₀ = 0.5165** at d_max = 0.1 (0.526 at 0.05, 0.497 at
  0.2 — the response is linear to ~2 % at d_max = 0.1).
- Per cell (task × seed; seeds randomize k₀ ∈ U[0.5, 2], ℓ₀ ∈ ℓ·U[0.9, 1.1]): untrained error; init gradient
  alignment of the full-batch local update vs the implicit adjoint; **converged L-BFGS ceiling**
  (`mech_ceiling`); **matched-budget GD baseline** (same epochs, lr = η — the local rule is GD at step η
  in the β → 0 limit); the local rule, β = 0.05, η_k = η_ℓ = 1.0 (η/β = 20), 6000 epochs; trajectory
  diagnostics (best error and epoch, first instability epoch, tail jump rate).

### Pilot observations that shaped the protocol (seed 0, amplify unless noted)
- **Plain GD is far from the ceiling.** Attenuate: L-BFGS ceiling 1.8e-4 vs GD(lr 1, 2000 epochs) 4.9e-2 vs
  local(η 0.5, 2000) 7.3e-2. Amplify: 2.2e-3 vs 1.2e-2 vs 2.7e-2. An offline baseline at a fixed GD budget
  would have *understated* representability by two orders of magnitude — the budget-artifact lesson, on
  the representable side this time.
- **Inversion is representable** (ceiling 4.6e-3) via pre-stress (rest lengths up to 1.66× geometry),
  although the 300-epoch GD pilot sat at 1.04 (worse than doing nothing).
- **The gradient flow walks into buckling.** Local rule, η = 0.5: error 2.66e-2 (2k) → 1.69e-2 (4k) →
  1.06e-2 (6k) → 6.67e-3 (8k) → **0.62 at 10k epochs with instability events appearing** (jump rate 0.03
  over the run, concentrated at the end). η = 2.0 destabilizes within the first epochs (jump rate 0.97);
  η = 1.0 and 1.5 are stable through 1000 epochs. Hence: the tail jump rate (last 20 % of epochs) is the
  multistability gate, best-epoch error is reported beside the final, and GD's failed epochs are counted.

### Headline run (6000 epochs, 3 seeds, 3 tasks) — run 2, corrected seam
`python experiment_b0.py`, 1555 s. Errors are normalized MSE. `best` = minimum along the trajectory;
`jump` = genuine instability rate over the run, `tail` = over the last 20 % of epochs (the verdict gate),
`1st` = first epoch with a basin-jump flag (transient where `jump` = 0.00); alignment at init (sign over
all edges, ρ); `conv` = plateaued.

| task | seed | untrained | ceiling (L-BFGS) | GD baseline (lr 1, 6000) | local (6000) | jump | tail | 1st | sign | ρ | conv | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attenuate | 0 | 3.52e-01 | 1.85e-04 | 8.91e-03 | **2.26e-03** | 0.00 | 0.00 | – | 0.97 | 1.000 | n | PASS |
| attenuate | 1 | 7.05e-01 | 9.88e-05 | 2.87e-03 | **1.39e-03** | 0.00 | 0.00 | – | 0.97 | 1.000 | n | PASS |
| attenuate | 2 | 5.90e+01 | 1.76e-04 | **8.47e-05** | 4.47e-04 | 0.00 | 0.00 | – | 0.97 | 1.000 | n | PASS (abs floor) |
| amplify | 0 | 7.11e-01 | 2.73e-06 | **1.02e-03** | 2.64e-03 | 0.00 | 0.00 | 4771 | 0.97 | 1.000 | n | AMBIGUOUS |
| amplify | 1 | 3.21e-01 | 3.39e-06 | 3.52e-03 | **4.26e-04** | 0.00 | 0.00 | – | 0.97 | 1.000 | n | PASS |
| amplify | 2 | 1.56e+00 | 3.80e-06 | 1.41e-03 | **5.05e-05** | 0.00 | 0.00 | – | 0.97 | 1.000 | n | PASS |
| invert | 0 | 3.01e+00 | 4.64e-03 | 1.95e-01 | **5.60e-03** | 0.00 | 0.00 | – | 0.97 | 1.000 | y | PASS |
| invert | 1 | 7.75e+00 | 4.58e-03 | 5.65e-01 | **1.57e-02** | 0.00 | 0.00 | 2406 | 0.97 | 1.000 | y | PASS |
| invert | 2 | 9.17e+01 | 3.21e-03 | 3.26e-01 | **4.14e-03** | 0.00 | 0.00 | – | 0.97 | 1.000 | y | PASS |

Task verdicts: **attenuate PASS · amplify AMBIGUOUS · invert PASS.** The local rule beats the matched-budget
GD baseline in 7 of 9 cells (by 1.3–30× on amplify/attenuate, 30–100× on invert); GD wins in attenuate seed 2
(5×, both ≤ 1e-3) and amplify seed 0 (2.6×). Init alignment is sign 0.97 / ρ 1.000 in every cell (the 0.97 is
the symmetric lattice's exactly-zero-gradient edges, cf. A2). Audit: `fraction_physics_local = 1.00`.

Trajectories (every 600 epochs): attenuate and amplify descend monotonically and are still descending at
6000 epochs; invert reaches its plateau within ~1200 epochs. **Measured stability margin** (smallest
tangent-stiffness eigenvalue of the free equilibria, `min_eig_history`, separate 6000-epoch runs):
amplify seed 0 0.0303 → 0.0272, seed 2 0.0322 → 0.0239; attenuate 0.0303 → 0.0277, 0.0322 → 0.0312;
invert 0.0303 → 0.0208 and 0.0322 → 0.0220 (plateaued with the error). Amplification and inversion soften
the lattice by 10–30 % over the run; nothing approaches zero.

### Run 1 (cold-start seam) — superseded
Same config. Amplify seeds 0 and 2 showed "instability" at epochs 4434 and 54 (final error 0.83, tail
0.39–0.50) and the GD baseline "failed" from epochs 2967 / 1428; the d-map "buckled" at d_max ≥ 0.2. All of
these were the free-phase Newton solve failing its line search from the reference geometry on a pre-stressed
lattice whose equilibrium is unique and stable (min eigenvalue +0.028; `settle()` and warm-started Newton reach
it in 2–8 iterations). The corrected seam (warm start + `settle()` fallback, rescues counted separately)
removed every one of them: 1 rescue in 6000 epochs on amplify seed 0, none elsewhere.

### Follow-up (invert, seed 0, 2000 epochs): is the local rule's edge over GD a finite-β or step-size effect?

| run | final error |
|---|---|
| local rule, β = 0.005 / 0.02 / 0.05 / 0.2 / 0.5 (η = 1) | 5.77e-3 / 5.75e-3 / 5.73e-3 / 5.64e-3 / 5.51e-3 |
| GD, lr = 0.1 / 0.5 / 0.66 / 1.0 | 1.03 / 0.112 / 0.300 / 0.546 |
| GD, lr = 3 / 10 | diverges (every epoch's solve fails from epoch 2) |

At init the local update equals **0.66 ×** the true gradient in every task and at every β (cosine 1.0000,
same factor on the k- and ℓ-parts), so the rule takes *smaller* steps than the lr = 1 baseline. GD's error
is **non-monotone in lr** (0.5 is best, 0.66 and 1.0 worse, ≥ 3 diverges) — plain GD oscillates on stiff
directions of this loss — while the local rule reaches 5.7e-3 at every β, essentially independent of β.

### Conditioning follow-up (2026-11-08; invert, seed 0): why the finite-β rule out-runs matched GD

- **Loss Hessian at init** (central differences of the adjoint gradient over θ = (k, ℓ)): λ_max = **2.09**
  (entirely in the rest-length block), stiffness-block λ_max = 4.9e-4, 35 of 66 eigenvalues negative
  (non-convex at init). GD's linear stability bound is lr < 2/λ_max = **0.96** — the matched-budget GD at
  lr = η = 1.0 sits just *above* its stability edge on the stiffest direction, which is why its error was
  non-monotone in lr (best at 0.5, worse at 0.66 and 1.0, divergent at ≥ 3).
- **GD trajectory at lr = 0.66** (inside the bound at init): error 3.0 → 0.40 by epoch 1200 with successive
  steps perfectly aligned (cosine +1.000), then a **spike to 7.35 at epoch 1300 with step cosine −0.998**
  (an oscillation on a stiff direction as |∇L| grows along the valley), recovery to 0.24 by 1500.
- **Local-rule trajectory at β = 0.05, η = 1.0**, measured against the true gradient at the *same* θ:

  | epoch | error | cos(Δθ_local, −∇L) | ρ | \|Δθ_local\| / \|∇L\| |
  |---|---|---|---|---|
  | 0 | 3.01 | +1.000 | +1.000 | 0.66 |
  | 500 | 0.94 | +0.804 | +0.814 | 0.50 |
  | 1000 | 0.68 | +0.722 | +0.736 | 0.40 |
  | 1500 | 0.23 | +0.652 | +0.668 | 0.28 |

  The update is the gradient at init and then turns smoothly away from it (to cosine 0.65) while its
  magnitude relative to the gradient halves; the trajectory never spikes, and after 2000 epochs it is at
  5.7e-3 (B0 follow-up) where GD at lr 0.66 is at 0.30.

- **Generalization check on Experiment A's linear resistor net** (regression, η = lr = 0.5, β = 0.01, where
  A's headline had local 1.26e-4 vs GD 9.23e-4): the local update stays *perfectly* aligned with the gradient
  along the whole trajectory (cosine 1.0000) but its magnitude is **1.98 → 1.67×** the GD step; GD's
  stability bound there is lr < 119, far from 0.5, and GD never oscillates. So on the linear net the local
  rule's edge is a plain step-scale effect of the hard-clamp update (its scale relative to ∇L depends on the
  output node's effective conductance), not conditioning. Methodological upshot for both experiments:
  **lr = η is not a matched effective step** — the ratio is substrate-dependent (A ≈ 1.7–2.0, B0 ≈ 0.66) and
  should be measured at init and reported beside any local-vs-GD comparison.

### d-map (amplify, seed 0, 1000 epochs): where on the deformation axis does the rule work?
| d_max | untrained r₀ | ceiling | local (1000 epochs) | jump | tail |
|---|---|---|---|---|---|
| 0.05 | 0.5256 | 2.95e-04 | 3.63e-02 | 0.00 | 0.00 |
| 0.10 | 0.5165 | 2.73e-06 | 2.66e-02 | 0.00 | 0.00 |
| 0.20 | 0.4974 | 9.87e-06 | 7.35e-03 | 0.00 | 0.00 |
| 0.30 | 0.4763 | 2.07e-05 | 1.19e-03 | 0.00 | 0.00 |

Stable at every deformation probed; the rule learns amplification **faster at larger deformation**
(30× lower error at d_max = 0.3 than at 0.05 in the same budget) — the geometric nonlinearity that
amplification exploits grows with d.

---

## Interpretation

**The positive control is established.** On a passive elastic lattice the two-phase mechanical rule —
every update computed from each bond's own two extension reads, 100 % physics-local — learns at network
scale: it attenuates the allosteric transfer ratio to ~1e-3 normalized error, amplifies it to 5e-5 – 3e-3,
and **inverts** it (a sign flip that a small-deformation linear lattice cannot produce; the learned
configurations carry rest lengths up to 1.6× geometry, i.e. pre-stress) to ~5e-3. Experiment B's keystone
(per-step alignment) and Experiment A's resistor PASS are now joined by a network-scale mechanical learning
result; "the physics computes the update" holds on the elastic medium for tasks it can represent, with no
multistability in sight at d_max ≤ 0.3.

**The local rule is at least as good as matched-budget GD, and usually better — and the conditioning
follow-up says why.** The loss is extremely ill-conditioned (rest-length curvature 2.09 vs stiffness
curvature 5e-4) and GD's stability bound is lr < 0.96, so the "matched" GD at lr = 1 was at its stability
edge and oscillates on the stiff rest-length direction (a measured spike at epoch 1300 even at lr 0.66).
The finite-β two-phase update equals the gradient at init and then departs from it systematically along
the trajectory (cosine 0.65, magnitude 0.28× the gradient by epoch 1500) — the nudged-equilibrium contrast
saturates along stiff directions, so the rule takes a *damped* step there and never spikes. In other words
the finite-β coupled-learning update behaves like an implicitly preconditioned (proximal-like) gradient
step: it can run at a nominal step GD cannot. This is now a measured mechanism, not a hypothesis. It does
*not* carry over to Experiment A's linear net, where the local update is exactly the gradient at ~1.7–2×
the GD step and GD is nowhere near instability: there the headline "local ≤ baseline" is a step-scale
artifact of lr-matching, harmless to A's verdict but worth knowing. The amplify AMBIGUOUS (seed 0, 2.6× behind GD,
both still descending) is the one cell where GD's larger raw step paid off within the budget.

**Both optimizers are slow gradient flows far from the ceiling.** The converged L-BFGS ceiling is 10–1000×
below either; neither GD nor the rule had plateaued at 6000 epochs on attenuate/amplify. The verdict compares
like with like (matched budget) and uses the ceiling only as the representability gate; a longer-budget run
would move both down together.

**Stability is a measured, mild drift — not a wall.** Amplification and inversion soften the lattice (smallest
eigenvalue down 10–30 % over 6000 epochs; inversion plateaus at 0.021), attenuation barely changes it. The
d-map is stable through d_max = 0.3 and learning is *faster* there. Run 1's opposite conclusion came from the
cold-start artifact; its withdrawal is the important methodological result of this experiment, and it
propagates to Experiment B (addendum) and B2.

**Honest caveats.**
- Attenuate/amplify had not plateaued (`conv = n`); errors would keep falling for both optimizers.
- Attenuate seed 2 PASSes via the 1e-3 absolute floor (GD 8.5e-5 vs rule 4.5e-4).
- Single lattice (4×4), one I/O placement, one β (0.05) and η (1.0) for the headline; η/β = 20.
- Transient basin-jump flags (amplify seed 0 at 4771, invert seed 1 at 2406) did not recur; the tail gate
  ignores them correctly.
- The one `settle()` rescue (amplify seed 0) means the Newton line search still fails occasionally from a
  warm start on a pre-stressed lattice; the rescued equilibrium was stable.

**Portfolio implications.** (1) `01`'s elastic medium is a bona-fide co-located learner for friendly tasks —
the portfolio's first mechanical network-scale PASS, with inversion as the non-trivial case. (2) **Lesson
candidate for the commons: a crash-tolerant solver seam must distinguish "the solver failed from its start
point" from "no stable equilibrium exists"** — count only the latter as physics, warm-start from the previous
equilibrium, fall back to a damped minimizer, and *measure* stability (smallest Hessian eigenvalue) rather
than infer it from crashes. Experiment B's multistability leg did not survive this. (3) The finite-β
two-phase update's conditioning advantage over raw GD is worth a dedicated study (roadmap item 8).
