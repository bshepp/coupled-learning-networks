# Coupled-Learning Networks

Decentralized, **physics-driven** learning in passive resistor/flow networks. The question this
project answers: *when a network adapts to a task, **who computes the weight update?*** If a CPU
optimizer does it, the substrate is a peripheral. The goal is a rule where every weight change is
derivable from quantities **physically local** to each element — the substrate's own physics performs
the update.

This is project `01` (the flagship) of the **physical-learning-substrates** portfolio, branching from
the co-located ionic computing line.

## Thesis under test

A learning substrate where *the signal being processed is the same physical event that reshapes the
processor* — compute and weight-update co-located. **Experiment A** (this repo's first milestone)
demonstrates the mechanism on *friendly* tasks a passive positive-conductance network can represent:

- **Allostery** — train edge conductances so a source drive produces a target response at distant nodes
  (Stern *PRX* 2021; Dillavou *PRApplied* 2022).
- **Regression** — learn a representable linear input→output map.

The verdict is twofold: the local **coupled-learning** rule must (1) reach the offline optimizer's task
error, and (2) produce updates that align with the true loss gradient as the nudge `β→0` (the
equilibrium-propagation limit). An **audit hook** logs that every headline update is physics-local.

> Experiment B (signed/non-monotone elastic networks attempting parity-3) is deferred to its own spec —
> it needs a vector nonlinear solver and faces a multistability risk. Positive conductances here share
> the maximum-principle wall, so Experiment A is deliberately **not** for parity.

## Structure

| File | Responsibility |
|---|---|
| `solver.py` | Batched Newton/Laplacian equilibrium solve + the `relax` seam |
| `network.py` | Constitutive laws (resistor, Hagen–Poiseuille), `Network`, builders |
| `coupled_learning.py` | The two-phase coupled-learning rule (the heart) |
| `tasks.py` | Allostery + regression tasks, error metric, verdict |
| `baselines.py` | Offline adjoint-gradient baseline + gradient-alignment stats |
| `audit.py` | The "who computes the update?" provenance hook |
| `experiment_a.py` | Driver: train, compare, emit verdict + JSON |
| `docs/superpowers/` | Design spec and implementation plan |

## Running

```bash
pip install -r requirements.txt
python -m pytest -q          # the test suite
python experiment_a.py       # reproduce the headline verdict (prints a table + writes JSON)
```

**Stack:** Python, numpy, scipy, pytest. The true gradient is a hand-written adjoint (no deep-learning
framework), cross-checked against finite differences.

## Method discipline

Compute/simulation first; foreclosure is a valid, cheap outcome; observation is kept separate from
interpretation; the offline-optimizer baseline is reported every time; pass/fail criteria are
pre-registered. See `docs/superpowers/specs/` for the design and `docs/superpowers/plans/` for the plan.

## References

- Stern, Hexner, Rocks, Liu — *PRX* 11, 021045 (2021), "Supervised learning in physical networks."
- Dillavou, Stern, Liu, Durian — *Phys. Rev. Applied* 18, 014040 (2022), "Demonstration of
  decentralized, physics-driven learning."
- Stern & Murugan — *Annu. Rev. Condens. Matter Phys.* 14 (2023), "Learning without neurons."
- Scellier & Bengio (2017), "Equilibrium propagation."
