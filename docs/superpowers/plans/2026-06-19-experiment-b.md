# Experiment B (elastic parity-3 escape) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 2D nonlinear spring-network substrate and test whether signed/non-monotone elastic state, trained by a local physics-driven coupled-learning rule, can separate parity-3 — the bet that the parity wall is about passivity, not co-located learning.

**Architecture:** Approach 1 — parallel mechanical modules in the existing `coupled-learning-networks` repo (Experiment A untouched; `tasks.py`/`baselines.py` extended additively). New: `spring_network.py` (energy/force/tangent-stiffness + smoothed tension-compression asymmetry + triangulated builder), `mechanical_solver.py` (damped-Newton relax + basin-jump), `mechanical_learning.py` (two-phase local `k`/`ℓ` update), `experiment_b.py` (driver). Reuses `audit.py` and `gradient_alignment`.

**Tech Stack:** Python 3, numpy, pytest. No jax/torch (the implicit adjoint is hand-written and cross-checked against finite differences).

## Global Constraints

- **Stack:** numpy + pytest only (scipy permitted but unused). No deep-learning frameworks.
- **Layout:** flat modules at repo root; tests in `tests/`. Root `conftest.py` already puts the repo root on `sys.path`.
- **House pattern:** small focused modules; functions return numbers/arrays; runnable modules have a `main()` printing a table + dumping JSON; one pytest module per source module.
- **Constitutive law:** bond energy `E = ½ k·m(e)·e²`, `e = L − ℓ`, `L = |xi − xj|`. Smoothed asymmetry multiplier `m(e) = α + (1−α)·½(1+tanh(e/w))`; `α=1 ⇒ m≡1` (symmetric, primary). Keep everything `C∞`.
- **Contrastive convention (consistent with A):** `Δθ = (η/β)[∂E_free/∂θ − ∂E_nudged/∂θ]`, edge-local. `∂E/∂k = ½ m(e)e²`; `∂E/∂ℓ = −k(½ m'(e)e² + m(e)e)`.
- **Passivity / bounds:** clip `k ∈ [k_min, k_max]` with `k_min>0` and `ℓ ∈ [ℓ_min, ℓ_max]` with `ℓ_min>0`, every step. `k>0` does NOT impose monotonicity here (geometry supplies non-monotonicity).
- **Multistability:** nudged phase warm-starts from the free equilibrium; `basin_jump` detects/log s basin changes; a high basin-jump rate is a distinct verdict, not a silent confound.
- **Verdict (five-way):** MULTISTABILITY-FORECLOSE (checked first) / FORECLOSE-representability / FORECLOSE-learnability / PASS (SNR≥k, cross-seed robust) / AMBIGUOUS.
- **Unknown outcome:** tests assert the pipeline emits a *valid* verdict — NEVER a specific one. The PASS/FORECLOSE is the experiment's result, recorded in a findings doc.
- **Audit:** every weight update tagged; headline coupled-learning runs assert 100% physics-local.
- **Commits:** every commit message ends with the trailer in Task 1 Step 5 (`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`).

---

## File structure

| File | Responsibility |
|---|---|
| `spring_network.py` | `SpringNetwork`; `m`/`dm`/`d2m`; `energy`, `forces`, `tangent_stiffness`; `dof`, `default_rest_lengths`, `bond_extensions`; `triangulated_lattice` |
| `mechanical_solver.py` | `relax_mech` (damped Newton + line search), `basin_jump` |
| `mechanical_learning.py` | `readout`, `free_phase_mech`, `nudged_phase_mech`, `local_update_mech`, `train_mech` |
| `tasks.py` (extend) | `parity3_mech_task`, `two_sided_separation`, `verdict_b` |
| `baselines.py` (extend) | `mech_adjoint_gradient`, `gd_baseline_mech` |
| `experiment_b.py` | `run`, `main` |
| `tests/test_*.py` | one per new/extended module |

---

## Task 1: Spring network core — energy & forces (`spring_network.py`)

**Files:**
- Create: `spring_network.py`
- Test: `tests/test_spring_network.py`

**Interfaces:**
- Produces: `m(e, alpha, w)`, `dm(e, alpha, w)`, `d2m(e, alpha, w)` (vectorized); `SpringNetwork(x0, edges, pinned_dofs, input_dofs, output_dof, alpha=1.0, w=0.1)` (dataclass; `x0` is `(N,2)`, `edges` list of `(i,j)`, the `_dofs` are flat indices `2*node+dim`); `dof(node, dim)`; `default_rest_lengths(net)` → `(E,)`; `bond_extensions(net, x, l)` → `(E,)` (`L−ℓ` per bond); `energy(net, x, k, l)` → float; `forces(net, x, k, l)` → `(N,2)` (`= −∂E/∂x`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spring_network.py
import numpy as np
import pytest
from spring_network import (SpringNetwork, m, dm, d2m, dof, default_rest_lengths,
                            bond_extensions, energy, forces)


def _net(alpha=1.0, w=0.2):
    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    edges = [(0, 1), (1, 2), (2, 0)]
    return SpringNetwork(x0, edges, pinned_dofs=[0, 1], input_dofs=[2],
                         output_dof=5, alpha=alpha, w=w)


def test_m_symmetric_is_one():
    e = np.array([-1.0, 0.0, 1.0])
    assert m(e, 1.0, 0.2) == pytest.approx([1.0, 1.0, 1.0])
    assert dm(e, 1.0, 0.2) == pytest.approx([0.0, 0.0, 0.0])
    assert d2m(e, 1.0, 0.2) == pytest.approx([0.0, 0.0, 0.0])


def test_m_derivatives_match_fd():
    e = np.array([-0.3, 0.1, 0.5]); alpha, w = 0.5, 0.2
    eps = 1e-6
    fd1 = (m(e + eps, alpha, w) - m(e - eps, alpha, w)) / (2 * eps)
    fd2 = (dm(e + eps, alpha, w) - dm(e - eps, alpha, w)) / (2 * eps)
    assert dm(e, alpha, w) == pytest.approx(fd1, abs=1e-5)
    assert d2m(e, alpha, w) == pytest.approx(fd2, abs=1e-4)


def test_default_rest_lengths_and_extensions():
    net = _net()
    l = default_rest_lengths(net)
    assert l == pytest.approx([1.0, np.hypot(0.5, 0.8), np.hypot(0.5, 0.8)])
    # at reference, extensions are zero
    assert bond_extensions(net, net.x0, l) == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_forces_match_negative_energy_gradient(alpha):
    net = _net(alpha=alpha)
    rng = np.random.default_rng(0)
    x = net.x0 + 0.1 * rng.standard_normal(net.x0.shape)
    k = np.array([1.0, 1.3, 0.8]); l = np.array([0.9, 1.0, 1.0])
    F = forces(net, x, k, l)
    eps = 1e-6
    fd = np.zeros_like(x)
    for i in range(x.shape[0]):
        for d in range(2):
            xp = x.copy(); xp[i, d] += eps
            xm = x.copy(); xm[i, d] -= eps
            fd[i, d] = -(energy(net, xp, k, l) - energy(net, xm, k, l)) / (2 * eps)
    assert F == pytest.approx(fd, abs=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spring_network.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spring_network'`.

- [ ] **Step 3: Write minimal implementation**

```python
# spring_network.py
"""2D nonlinear spring network: positions, the smoothed-asymmetry constitutive law, and
energy/force/tangent-stiffness. Bond energy E = 0.5*k*m(e)*e^2 with e = L - l, L = |xi - xj|.
m(e) is a smooth tension-compression multiplier (m==1 when alpha==1). The geometric
nonlinearity (through L and the rotating bond direction) is what makes the response non-monotone."""
import numpy as np
from dataclasses import dataclass
from typing import List


def m(e, alpha, w):
    """Smooth asymmetry multiplier: ~1 in tension, ~alpha in compression. alpha=1 -> 1."""
    e = np.asarray(e, float)
    return alpha + (1.0 - alpha) * 0.5 * (1.0 + np.tanh(e / w))


def dm(e, alpha, w):
    e = np.asarray(e, float)
    return (1.0 - alpha) * 0.5 / w / np.cosh(e / w) ** 2


def d2m(e, alpha, w):
    e = np.asarray(e, float)
    t = np.tanh(e / w)
    return (1.0 - alpha) * 0.5 * (-2.0 * t) / (w * w) / np.cosh(e / w) ** 2


@dataclass
class SpringNetwork:
    x0: np.ndarray              # (N, 2) reference positions
    edges: List[tuple]          # list of (i, j)
    pinned_dofs: List[int]      # flat DOF indices held at x0
    input_dofs: List[int]       # flat DOF indices driven by inputs
    output_dof: int             # flat DOF index of the scalar readout
    alpha: float = 1.0
    w: float = 0.2


def dof(node, dim):
    return 2 * node + dim


def default_rest_lengths(net):
    x = net.x0
    return np.array([np.linalg.norm(x[i] - x[j]) for (i, j) in net.edges], float)


def bond_extensions(net, x, l):
    x = np.asarray(x, float)
    L = np.array([np.linalg.norm(x[i] - x[j]) for (i, j) in net.edges], float)
    return L - np.asarray(l, float)


def energy(net, x, k, l):
    x = np.asarray(x, float)
    E = 0.0
    for b, (i, j) in enumerate(net.edges):
        L = np.linalg.norm(x[i] - x[j]); e = L - l[b]
        E += 0.5 * k[b] * float(m(e, net.alpha, net.w)) * e * e
    return float(E)


def forces(net, x, k, l):
    """F = -dE/dx, shape (N, 2). Per bond the scalar tension T = dE/dL = 0.5*k*m'(e)*e^2 + k*m(e)*e."""
    x = np.asarray(x, float)
    F = np.zeros_like(x)
    for b, (i, j) in enumerate(net.edges):
        d = x[i] - x[j]; L = np.linalg.norm(d); dh = d / L; e = L - l[b]
        T = 0.5 * k[b] * float(dm(e, net.alpha, net.w)) * e * e + k[b] * float(m(e, net.alpha, net.w)) * e
        F[i] -= T * dh; F[j] += T * dh
    return F
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spring_network.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add spring_network.py tests/test_spring_network.py
git commit -m "$(cat <<'EOF'
feat: spring network energy and forces

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Tangent stiffness & triangulated builder (`spring_network.py`)

**Files:**
- Modify: `spring_network.py`
- Test: `tests/test_spring_network.py` (append)

**Interfaces:**
- Consumes: `energy`, `forces`, `SpringNetwork` (Task 1).
- Produces: `tangent_stiffness(net, x, k, l)` → `(2N, 2N)` (the Hessian `∂²E/∂x²`); `triangulated_lattice(rows, cols, spacing=1.0)` → `(x0, edges)`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_tangent_stiffness_matches_fd():
    from spring_network import tangent_stiffness
    net = _net(alpha=0.5)
    rng = np.random.default_rng(1)
    x = net.x0 + 0.1 * rng.standard_normal(net.x0.shape)
    k = np.array([1.0, 1.2, 0.7]); l = np.array([0.9, 1.0, 1.05])
    H = tangent_stiffness(net, x, k, l)
    n = x.shape[0]; eps = 1e-6
    fd = np.zeros((2 * n, 2 * n))
    g = lambda xx: -forces(net, xx, k, l).reshape(-1)   # gradient of E
    for b in range(2 * n):
        xp = x.reshape(-1).copy(); xp[b] += eps
        xm = x.reshape(-1).copy(); xm[b] -= eps
        fd[:, b] = (g(xp.reshape(n, 2)) - g(xm.reshape(n, 2))) / (2 * eps)
    assert H == pytest.approx(fd, abs=1e-4)
    assert H == pytest.approx(H.T, abs=1e-9)          # symmetric


def test_triangulated_lattice_is_rigid():
    from spring_network import triangulated_lattice, tangent_stiffness, default_rest_lengths
    x0, edges = triangulated_lattice(3, 3)
    assert x0.shape == (9, 2)
    net = SpringNetwork(x0, edges, pinned_dofs=[0, 1, dof(2, 1)],   # pin a corner fully + one DOF
                        input_dofs=[], output_dof=0)
    k = np.ones(len(edges)); l = default_rest_lengths(net)
    H = tangent_stiffness(net, x0, k, l)
    free = [d for d in range(2 * 9) if d not in {0, 1, dof(2, 1)}]
    Hff = H[np.ix_(free, free)]
    assert np.linalg.matrix_rank(Hff) == len(free)    # non-singular -> rigid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spring_network.py -q`
Expected: FAIL with `ImportError: cannot import name 'tangent_stiffness'`.

- [ ] **Step 3: Write minimal implementation (append to `spring_network.py`)**

```python
def tangent_stiffness(net, x, k, l):
    """Hessian d^2 E / dx^2, shape (2N, 2N). Per bond: material term kt*(dh dh^T) plus geometric
    term (T/L)*(I - dh dh^T), assembled as [[K,-K],[-K,K]] over the bond's two nodes.
    kt = d^2 E/dL^2 = 0.5*k*(m''(e)*e^2 + 4*m'(e)*e + 2*m(e));  T = dE/dL."""
    x = np.asarray(x, float)
    n = x.shape[0]
    H = np.zeros((2 * n, 2 * n))
    I2 = np.eye(2)
    for b, (i, j) in enumerate(net.edges):
        d = x[i] - x[j]; L = np.linalg.norm(d); dh = d / L; e = L - l[b]
        me = float(m(e, net.alpha, net.w)); m1 = float(dm(e, net.alpha, net.w)); m2 = float(d2m(e, net.alpha, net.w))
        T = 0.5 * k[b] * m1 * e * e + k[b] * me * e
        kt = 0.5 * k[b] * (m2 * e * e + 4.0 * m1 * e + 2.0 * me)
        P = np.outer(dh, dh)
        K = kt * P + (T / L) * (I2 - P)
        for (a, sa) in ((i, +1), (j, -1)):
            for (c, sc) in ((i, +1), (j, -1)):
                H[2 * a:2 * a + 2, 2 * c:2 * c + 2] += (sa * sc) * K
    return H


def triangulated_lattice(rows, cols, spacing=1.0):
    """Rectangular grid of nodes with horizontal, vertical, and one diagonal per cell -> rigid."""
    idx = lambda r, c: r * cols + c
    x0 = np.array([[c * spacing, r * spacing] for r in range(rows) for c in range(cols)], float)
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
            if r + 1 < rows and c + 1 < cols:
                edges.append((idx(r, c), idx(r + 1, c + 1)))
    return x0, edges
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spring_network.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add spring_network.py tests/test_spring_network.py
git commit -m "$(cat <<'EOF'
feat: spring tangent stiffness and triangulated lattice builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Mechanical equilibrium solver (`mechanical_solver.py`)

**Files:**
- Create: `mechanical_solver.py`
- Test: `tests/test_mechanical_solver.py`

**Interfaces:**
- Consumes: `energy`, `forces`, `tangent_stiffness` (spring_network).
- Produces: `relax_mech(net, k, l, clamp, x_init=None, max_iter=100, tol=1e-9) -> (x, info)` where `clamp` is `{flat_dof: absolute_coord_value}`, `x` is `(N,2)`, `info` has `converged`, `iters`, `energy`; floating DOFs are those not in `clamp`. `basin_jump(x_free, x_nudged, expected, factor=10.0) -> (jumped: bool, magnitude: float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mechanical_solver.py
import numpy as np
import pytest
from spring_network import SpringNetwork, default_rest_lengths, dof
from mechanical_solver import relax_mech, basin_jump


def _bridge():
    # anchors (0,0) and (2,0) pinned; free node near (1,1); rest length sqrt(2) -> equilibrium (1,1)
    x0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    edges = [(0, 1), (1, 2)]
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(0, 0), dof(0, 1), dof(2, 0), dof(2, 1)],
                        input_dofs=[], output_dof=dof(1, 1))
    l = default_rest_lengths(net)   # both sqrt(2)
    return net, l


def test_relax_finds_known_equilibrium():
    net, l = _bridge()
    k = np.array([1.0, 1.0])
    clamp = {d: net.x0.reshape(-1)[d] for d in net.pinned_dofs}  # anchors at reference
    x_init = net.x0.copy(); x_init[1] = [1.0, 0.6]               # perturb free node, upper basin
    x, info = relax_mech(net, k, l, clamp, x_init=x_init)
    assert info["converged"]
    assert x[1] == pytest.approx([1.0, 1.0], abs=1e-6)


def test_warm_start_reproduces_equilibrium():
    net, l = _bridge()
    k = np.array([1.0, 1.0])
    clamp = {d: net.x0.reshape(-1)[d] for d in net.pinned_dofs}
    x1, _ = relax_mech(net, k, l, clamp, x_init=net.x0.copy())
    x2, _ = relax_mech(net, k, l, clamp, x_init=x1.copy())       # warm start from equilibrium
    assert x2 == pytest.approx(x1, abs=1e-8)


def test_basin_jump_detector():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    near = a + np.array([[0.0, 0.0], [0.0, 0.02]])
    far = np.array([[0.0, 0.0], [1.0, -1.0]])
    assert basin_jump(a, near, expected=0.05)[0] is False
    assert basin_jump(a, far, expected=0.05)[0] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mechanical_solver.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mechanical_solver'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mechanical_solver.py
"""Quasi-static equilibrium of a spring network by damped Newton (energy minimization) with a
backtracking line search. clamp fixes input/pinned/output DOFs (Dirichlet); the rest are free.
warm-start (x_init) keeps the nudged phase in the free phase's basin; basin_jump flags when a
relaxation lands in a different basin than expected."""
import numpy as np
from spring_network import energy, forces, tangent_stiffness


def relax_mech(net, k, l, clamp, x_init=None, max_iter=100, tol=1e-9):
    n = net.x0.shape[0]
    x = (net.x0.copy() if x_init is None else np.asarray(x_init, float).copy())
    for d, v in clamp.items():
        node, dim = divmod(d, 2); x[node, dim] = v
    free = np.array([d for d in range(2 * n) if d not in clamp], int)
    if free.size == 0:
        return x, {"converged": True, "iters": 0, "energy": energy(net, x, k, l)}
    for it in range(max_iter):
        R = forces(net, x, k, l).reshape(-1)[free]          # net force on free DOFs (want 0)
        if np.max(np.abs(R)) < tol:
            return x, {"converged": True, "iters": it, "energy": energy(net, x, k, l)}
        H = tangent_stiffness(net, x, k, l)
        Hff = H[np.ix_(free, free)]
        try:
            step = np.linalg.solve(Hff, R)                  # Newton: minimize E (grad = -R, Hess = Hff)
        except np.linalg.LinAlgError:
            raise RuntimeError("singular tangent stiffness (floppy mode?) in relax_mech")
        E0 = energy(net, x, k, l)
        xf = x.reshape(-1).copy()
        s = 1.0
        for _ in range(40):
            xt = xf.copy(); xt[free] += s * step
            if energy(net, xt.reshape(n, 2), k, l) < E0:
                break
            s *= 0.5
        xf[free] += s * step
        x = xf.reshape(n, 2)
    raise RuntimeError(f"relax_mech did not converge in {max_iter} iters "
                       f"(max|force|={float(np.max(np.abs(R))):.2e})")


def basin_jump(x_free, x_nudged, expected, factor=10.0):
    """Flag a basin change: if the largest node displacement between the two equilibria exceeds
    `factor` times the expected nudge magnitude, the nudged solve fell into a different basin."""
    disp = float(np.max(np.linalg.norm(np.asarray(x_nudged) - np.asarray(x_free), axis=1)))
    return (disp > factor * expected), disp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mechanical_solver.py -q`
Expected: PASS (3 passed). If `test_relax_finds_known_equilibrium` lands at `[1,-1]`, the perturbed start was in the wrong basin — confirm `x_init[1] = [1.0, 0.6]` (upper basin) is used.

- [ ] **Step 5: Commit**

```bash
git add mechanical_solver.py tests/test_mechanical_solver.py
git commit -m "$(cat <<'EOF'
feat: mechanical equilibrium solver (damped Newton + basin-jump)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Parity-3 task, separation & verdict (`tasks.py` extend)

**Files:**
- Modify: `tasks.py`
- Test: `tests/test_tasks_mech.py`

**Interfaces:**
- Consumes: `SpringNetwork`, `triangulated_lattice`, `default_rest_lengths`, `dof` (spring_network).
- Produces: `parity3_mech_task(rows=4, cols=4, d=0.3, seed=0) -> dict` with keys `net`, `patterns` (8×3 of ±1), `labels` (8 of ±1), `d`, `target_pos`, `target_neg`; `two_sided_separation(r, y) -> float`; `verdict_b(local_sep, baseline_sep, floor, jump_rate, sign_match, rho, k=3.0, jump_max=0.2, sm_min=0.9, rho_min=0.8) -> dict` with key `status` in {MULTISTABILITY-FORECLOSE, FORECLOSE-REPRESENTABILITY, FORECLOSE-LEARNABILITY, PASS, AMBIGUOUS}.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks_mech.py
import numpy as np
import pytest
from tasks import parity3_mech_task, two_sided_separation, verdict_b


def test_parity3_task_structure():
    task = parity3_mech_task()
    assert task["patterns"].shape == (8, 3)
    assert set(np.unique(task["patterns"])) <= {-1.0, 1.0}
    assert task["labels"].shape == (8,)
    # parity label = product of bits
    assert task["labels"] == pytest.approx(np.prod(task["patterns"], axis=1))
    # input/output DOFs are distinct and exist
    net = task["net"]
    assert len(net.input_dofs) == 3
    assert net.output_dof not in net.input_dofs
    assert net.output_dof not in net.pinned_dofs


def test_two_sided_separation():
    y = np.array([1, 1, -1, -1])
    assert two_sided_separation(np.array([2.0, 3.0, 0.0, 1.0]), y) == pytest.approx(1.0)   # +class above
    assert two_sided_separation(np.array([0.0, 1.0, 2.0, 3.0]), y) == pytest.approx(1.0)   # either polarity
    assert two_sided_separation(np.array([0.0, 2.0, 1.0, 3.0]), y) < 0                     # overlap


def test_verdict_b_branches():
    # multistability gate fires first regardless of separation
    assert verdict_b(1.0, 1.0, 0.1, jump_rate=0.5, sign_match=1.0, rho=1.0)["status"] == "MULTISTABILITY-FORECLOSE"
    # baseline can't separate -> representability foreclosure
    assert verdict_b(-0.1, -0.1, 0.1, 0.0, 1.0, 1.0)["status"] == "FORECLOSE-REPRESENTABILITY"
    # baseline separates, local rule can't -> learnability foreclosure
    assert verdict_b(-0.1, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "FORECLOSE-LEARNABILITY"
    # strong separation, aligned, no jumps -> PASS
    assert verdict_b(1.0, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "PASS"
    # positive but sub-SNR -> AMBIGUOUS
    assert verdict_b(0.15, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "AMBIGUOUS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks_mech.py -q`
Expected: FAIL with `ImportError: cannot import name 'parity3_mech_task'`.

- [ ] **Step 3: Write minimal implementation (append to `tasks.py`)**

```python
# --- Experiment B: mechanical parity-3 ---------------------------------------
from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof


def parity3_mech_task(rows=4, cols=4, d=0.3, seed=0):
    """Triangulated lattice; 3 input nodes on the bottom row (driven in y by +/-d), output node
    top-right (read y-displacement), bottom-left corner pinned fully + one more DOF to remove
    rigid-body motion. Targets span the achievable readout range for the nudge."""
    x0, edges = triangulated_lattice(rows, cols)
    in_nodes = [0 * cols + 0, 0 * cols + (cols // 2), 0 * cols + (cols - 1)]  # bottom row
    out_node = (rows - 1) * cols + (cols - 1)                                 # top-right
    pin_node = 0 * cols + 1                                                   # a bottom node not an input
    input_dofs = [dof(nd, 1) for nd in in_nodes]                              # drive in y
    pinned_dofs = [dof(pin_node, 0), dof(pin_node, 1), dof(in_nodes[0], 0)]   # 2 + 1 DOFs
    net = SpringNetwork(x0, edges, pinned_dofs=pinned_dofs, input_dofs=input_dofs,
                        output_dof=dof(out_node, 1))
    patterns = np.array([[1 - 2 * ((i >> b) & 1) for b in range(3)] for i in range(8)], float)
    labels = np.prod(patterns, axis=1)
    return {"net": net, "patterns": patterns, "labels": labels, "d": d,
            "target_pos": +0.5 * d, "target_neg": -0.5 * d}


def two_sided_separation(r, y):
    """Separable in EITHER polarity: max(min(+)-max(-), min(-)-max(+)). >0 iff a threshold splits
    the classes; <=0 iff they overlap. (Ported idiom from self-assembly/stage1_network.py.)"""
    r = np.asarray(r, float); y = np.asarray(y)
    pos, neg = r[y > 0], r[y < 0]
    return float(max(np.min(pos) - np.max(neg), np.min(neg) - np.max(pos)))


def verdict_b(local_sep, baseline_sep, floor, jump_rate, sign_match, rho,
              k=3.0, jump_max=0.2, sm_min=0.9, rho_min=0.8):
    """Five-way verdict, evaluated in order: multistability gate, then representability,
    learnability, PASS (SNR>=k and aligned and low jumps), else AMBIGUOUS."""
    snr = local_sep / floor if floor > 0 else float("inf")
    base_snr = baseline_sep / floor if floor > 0 else float("inf")
    out = {"local_sep": float(local_sep), "baseline_sep": float(baseline_sep),
           "snr": float(snr), "jump_rate": float(jump_rate),
           "sign_match": float(sign_match), "rho": float(rho)}
    if jump_rate > jump_max:
        out["status"] = "MULTISTABILITY-FORECLOSE"
    elif baseline_sep <= 0 or base_snr < 1.0:
        out["status"] = "FORECLOSE-REPRESENTABILITY"
    elif local_sep <= 0 or snr < 1.0:
        out["status"] = "FORECLOSE-LEARNABILITY"
    elif snr >= k and sign_match >= sm_min and rho >= rho_min:
        out["status"] = "PASS"
    else:
        out["status"] = "AMBIGUOUS"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks_mech.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tasks.py tests/test_tasks_mech.py
git commit -m "$(cat <<'EOF'
feat: mechanical parity-3 task, two-sided separation, verdict_b

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: The mechanical coupled-learning rule (`mechanical_learning.py`)

**Files:**
- Create: `mechanical_learning.py`
- Test: `tests/test_mechanical_learning.py`

**Interfaces:**
- Consumes: `relax_mech`, `basin_jump` (mechanical_solver); `bond_extensions`, `m`, `dm`, `energy` (spring_network); `PHYSICS_LOCAL`, `AuditLog` (audit); `two_sided_separation` (tasks).
- Produces: `readout(net, x) -> float` (output-DOF displacement from reference); `free_phase_mech(net, k, l, pattern, d) -> (x, info)`; `nudged_phase_mech(net, k, l, pattern, d, target, beta, x_free) -> (x, info, r_free)`; `local_update_mech(net, k, l, x_free, x_nudged, beta, eta_k, eta_l, audit=None) -> (dk, dl)`; `train_mech(task, k0, l0, beta, eta_k, eta_l, epochs, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None) -> dict` with keys `k`, `l`, `sep_history`, `jump_rate`, `readouts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mechanical_learning.py
import numpy as np
import pytest
from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof
from mechanical_learning import readout, free_phase_mech, nudged_phase_mech, train_mech
from audit import AuditLog


def _small_task():
    x0, edges = triangulated_lattice(3, 3)
    in_nodes = [0, 1, 2]
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(1, 0), dof(1, 1), dof(0, 0)],
                        input_dofs=[dof(n, 1) for n in in_nodes], output_dof=dof(8, 1))
    l = default_rest_lengths(net)
    k = np.ones(len(edges))
    return net, k, l


def test_nudge_moves_readout_toward_target():
    net, k, l = _small_task()
    pattern = np.array([1.0, -1.0, 1.0])
    x_free, _ = free_phase_mech(net, k, l, pattern, d=0.3)
    r_free = readout(net, x_free)
    target = r_free + 0.5         # push readout up
    x_nud, _, r0 = nudged_phase_mech(net, k, l, pattern, d=0.3, target=target, beta=0.2, x_free=x_free)
    assert r0 == pytest.approx(r_free, abs=1e-9)
    assert readout(net, x_nud) > r_free          # moved toward the (higher) target


def test_training_reduces_separation_gap_and_is_physics_local():
    # A friendly 2-pattern target: drive the readout of pattern A up, pattern B down (representable).
    net, k0, l0 = _small_task()
    patterns = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    labels = np.array([1.0, -1.0])
    task = {"net": net, "patterns": patterns, "labels": labels, "d": 0.3,
            "target_pos": 0.15, "target_neg": -0.15}
    audit = AuditLog()
    res = train_mech(task, k0, l0, beta=0.05, eta_k=2.0, eta_l=2.0, epochs=150, audit=audit)
    assert res["sep_history"][-1] > res["sep_history"][0]   # separation improved
    assert audit.assert_physics_local() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mechanical_learning.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mechanical_learning'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mechanical_learning.py
"""The two-phase mechanical coupled-learning rule (the heart of Experiment B). Free phase clamps
the input DOFs; the nudged phase additionally clamps the output readout DOF a small step toward the
target and is warm-started from the free equilibrium. Each bond updates its stiffness k and rest
length l from the LOCAL extension contrast between the two relaxed states. No global signal touches
a weight."""
import numpy as np
from spring_network import bond_extensions, m, dm
from mechanical_solver import relax_mech, basin_jump
from tasks import two_sided_separation
from audit import PHYSICS_LOCAL


def _input_clamp(net, pattern, d):
    clamp = {dd: net.x0.reshape(-1)[dd] for dd in net.pinned_dofs}
    for bit, dd in zip(pattern, net.input_dofs):
        node, dim = divmod(dd, 2)
        clamp[dd] = net.x0[node, dim] + float(bit) * d
    return clamp


def readout(net, x):
    node, dim = divmod(net.output_dof, 2)
    return float(x[node, dim] - net.x0[node, dim])


def free_phase_mech(net, k, l, pattern, d):
    return relax_mech(net, k, l, _input_clamp(net, pattern, d), x_init=net.x0)


def nudged_phase_mech(net, k, l, pattern, d, target, beta, x_free):
    r_free = readout(net, x_free)
    clamp = _input_clamp(net, pattern, d)
    node, dim = divmod(net.output_dof, 2)
    clamp[net.output_dof] = net.x0[node, dim] + (r_free + beta * (target - r_free))
    x, info = relax_mech(net, k, l, clamp, x_init=x_free)
    return x, info, r_free


def local_update_mech(net, k, l, x_free, x_nudged, beta, eta_k, eta_l, audit=None):
    ef = bond_extensions(net, x_free, l)
    en = bond_extensions(net, x_nudged, l)
    mf = m(ef, net.alpha, net.w); mn = m(en, net.alpha, net.w)
    dk = (eta_k / beta) * (0.5 * mf * ef ** 2 - 0.5 * mn * en ** 2)
    # dE/dl = -k*(0.5*m'(e)*e^2 + m(e)*e); contrast free - nudged
    dEdl_f = -k * (0.5 * dm(ef, net.alpha, net.w) * ef ** 2 + mf * ef)
    dEdl_n = -k * (0.5 * dm(en, net.alpha, net.w) * en ** 2 + mn * en)
    dl = (eta_l / beta) * (dEdl_f - dEdl_n)
    if audit is not None:
        audit.record(PHYSICS_LOCAL, 2 * len(net.edges))   # k and l per bond
    return dk, dl


def train_mech(task, k0, l0, beta, eta_k, eta_l, epochs,
               k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    net, patterns, labels, d = task["net"], task["patterns"], task["labels"], task["d"]
    targets = np.where(labels > 0, task["target_pos"], task["target_neg"])
    k = np.array(k0, float); l = np.array(l0, float)
    sep_history = []
    jumps = 0; trials = 0
    readouts = None
    for _ in range(epochs):
        dk = np.zeros_like(k); dl = np.zeros_like(l)
        readouts = np.zeros(len(patterns))
        for p in range(len(patterns)):
            xf, _ = free_phase_mech(net, k, l, patterns[p], d)
            readouts[p] = readout(net, xf)
            xn, _, r_free = nudged_phase_mech(net, k, l, patterns[p], d, targets[p], beta, xf)
            jumped, _ = basin_jump(xf, xn, expected=abs(beta * (targets[p] - r_free)) + 1e-12)
            jumps += int(jumped); trials += 1
            ddk, ddl = local_update_mech(net, k, l, xf, xn, beta, eta_k, eta_l, audit=audit)
            dk += ddk; dl += ddl
        k = np.clip(k + dk / len(patterns), k_min, k_max)
        l = np.clip(l + dl / len(patterns), l_min, l_max)
        sep_history.append(two_sided_separation(readouts, labels))
    return {"k": k, "l": l, "sep_history": sep_history,
            "jump_rate": jumps / max(trials, 1), "readouts": readouts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mechanical_learning.py -q`
Expected: PASS (2 passed). If `test_training_reduces_separation_gap...` does not improve, raise `epochs`/`eta_k`/`eta_l` (do NOT change the rule or its sign). If separation never improves and the readouts barely move, report it (possible representability limit of the tiny net) rather than forcing it.

- [ ] **Step 5: Commit**

```bash
git add mechanical_learning.py tests/test_mechanical_learning.py
git commit -m "$(cat <<'EOF'
feat: mechanical coupled-learning rule (free/nudged phases, k and l update)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implicit-adjoint gradient & baseline (`baselines.py` extend)

**Files:**
- Modify: `baselines.py`
- Test: `tests/test_baselines_mech.py`

**Interfaces:**
- Consumes: `relax_mech` (mechanical_solver); `tangent_stiffness`, `bond_extensions`, `m`, `dm` (spring_network); `free_phase_mech`, `readout` (mechanical_learning); `OFFLINE_GRADIENT` (audit); `two_sided_separation` (tasks).
- Produces: `mech_adjoint_gradient(net, k, l, patterns, d, targets) -> (dL_dk, dL_dl)` for `L = 0.5*mean_p (r_p - target_p)^2`; `gd_baseline_mech(task, k0, l0, epochs, lr, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None) -> dict` with keys `k`, `l`, `sep`, `readouts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines_mech.py
import numpy as np
import pytest
from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof
from mechanical_learning import free_phase_mech, readout
from baselines import mech_adjoint_gradient, gd_baseline_mech


def _setup():
    x0, edges = triangulated_lattice(3, 3)
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(1, 0), dof(1, 1), dof(0, 0)],
                        input_dofs=[dof(0, 1), dof(2, 1), dof(6, 1)], output_dof=dof(8, 1))
    l = default_rest_lengths(net); k = np.linspace(0.8, 1.2, len(edges))
    return net, k, l


def _fd_grad(net, k, l, patterns, d, targets):
    eps = 1e-6
    def loss(kk, ll):
        tot = 0.0
        for p in range(len(patterns)):
            xf, _ = free_phase_mech(net, kk, ll, patterns[p], d)
            tot += (readout(net, xf) - targets[p]) ** 2
        return 0.5 * tot / len(patterns)
    gk = np.zeros_like(k)
    for b in range(len(k)):
        kp = k.copy(); kp[b] += eps; km = k.copy(); km[b] -= eps
        gk[b] = (loss(kp, l) - loss(km, l)) / (2 * eps)
    gl = np.zeros_like(l)
    for b in range(len(l)):
        lp = l.copy(); lp[b] += eps; lm = l.copy(); lm[b] -= eps
        gl[b] = (loss(k, lp) - loss(k, lm)) / (2 * eps)
    return gk, gl


def test_adjoint_matches_fd():
    net, k, l = _setup()
    patterns = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    targets = np.array([0.1, -0.1]); d = 0.3
    gk, gl = mech_adjoint_gradient(net, k, l, patterns, d, targets)
    fgk, fgl = _fd_grad(net, k, l, patterns, d, targets)
    assert gk == pytest.approx(fgk, abs=1e-4)
    assert gl == pytest.approx(fgl, abs=1e-4)


def test_gd_baseline_reduces_loss():
    net, k0, l0 = _setup()
    patterns = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    labels = np.array([1.0, -1.0])
    task = {"net": net, "patterns": patterns, "labels": labels, "d": 0.3,
            "target_pos": 0.1, "target_neg": -0.1}
    res = gd_baseline_mech(task, k0, l0, epochs=120, lr=1.0)
    assert res["sep"] > -0.1          # baseline drives the readouts apart (or at least not worse)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines_mech.py -q`
Expected: FAIL with `ImportError: cannot import name 'mech_adjoint_gradient'`.

- [ ] **Step 3: Write minimal implementation (append to `baselines.py`)**

```python
# --- Experiment B: implicit-adjoint gradient through the nonlinear equilibrium ---------------
from spring_network import tangent_stiffness, bond_extensions, m as _mB, dm as _dmB, d2m as _d2mB


def mech_adjoint_gradient(net, k, l, patterns, d, targets):
    """dL/dk, dL/dl for L = 0.5*mean_p (r_p - t_p)^2 via the implicit function theorem.
    At equilibrium grad_x E = 0, so dx/dtheta = -Hff^{-1} d(grad_free E)/dtheta. Adjoint:
    Hff lam = dL/dx_free; dL/dtheta = -lam . d(grad_free E)/dtheta. lam = 0 on clamped DOFs."""
    from mechanical_learning import free_phase_mech, readout, _input_clamp
    n = net.x0.shape[0]
    gk = np.zeros(len(net.edges)); gl = np.zeros(len(net.edges))
    onode, odim = divmod(net.output_dof, 2)
    P = len(patterns)
    for p in range(P):
        x, _ = free_phase_mech(net, k, l, patterns[p], d)
        clamp = _input_clamp(net, patterns[p], d)
        free = np.array([dd for dd in range(2 * n) if dd not in clamp], int)
        fpos = {dd: idx for idx, dd in enumerate(free)}
        H = tangent_stiffness(net, x, k, l)
        Hff = H[np.ix_(free, free)]
        rhs = np.zeros(free.size)
        r = readout(net, x)
        rhs[fpos[net.output_dof]] = (r - targets[p]) / P          # dL/dx at the output free DOF
        lam = np.zeros(2 * n)
        lam[free] = np.linalg.solve(Hff, rhs)
        # d(grad E)/dtheta per bond: dgrad/dk_b at node i = (T_b/k_b) dh ; dgrad/dl_b = -kt_b dh
        for b, (i, j) in enumerate(net.edges):
            dvec = x[i] - x[j]; L = np.linalg.norm(dvec); dh = dvec / L; e = L - l[b]
            me = float(_mB(e, net.alpha, net.w)); m1 = float(_dmB(e, net.alpha, net.w))
            Tk = 0.5 * m1 * e * e + me * e                        # T/k = d(E/k)/dL coefficient
            dlam_i = lam[2 * i:2 * i + 2]; dlam_j = lam[2 * j:2 * j + 2]
            dgk = Tk * dh                                         # dgrad_i/dk ; node j = -dgk
            gk[b] += -(dlam_i @ dgk - dlam_j @ dgk)
            # kt for dl term
            m2 = float(_d2mB(e, net.alpha, net.w))
            kt = 0.5 * k[b] * (m2 * e * e + 4.0 * m1 * e + 2.0 * me)
            dgl = -kt * dh                                        # dgrad_i/dl ; node j = -dgl
            gl[b] += -(dlam_i @ dgl - dlam_j @ dgl)
    return gk, gl


def gd_baseline_mech(task, k0, l0, epochs, lr, k_min=1e-2, k_max=1e2, l_min=0.1, l_max=5.0, audit=None):
    from mechanical_learning import free_phase_mech, readout
    from audit import OFFLINE_GRADIENT
    from tasks import two_sided_separation
    net, patterns, labels, d = task["net"], task["patterns"], task["labels"], task["d"]
    targets = np.where(labels > 0, task["target_pos"], task["target_neg"])
    k = np.array(k0, float); l = np.array(l0, float)
    readouts = None
    for _ in range(epochs):
        gk, gl = mech_adjoint_gradient(net, k, l, patterns, d, targets)
        k = np.clip(k - lr * gk, k_min, k_max)
        l = np.clip(l - lr * gl, l_min, l_max)
        if audit is not None:
            audit.record(OFFLINE_GRADIENT, 2 * len(net.edges))
    readouts = np.array([readout(net, free_phase_mech(net, k, l, patterns[p], d)[0])
                         for p in range(len(patterns))])
    return {"k": k, "l": l, "sep": two_sided_separation(readouts, labels), "readouts": readouts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines_mech.py -q`
Expected: PASS (2 passed). If `test_adjoint_matches_fd` fails by a global sign on either `gk` or `gl`, flip that term's sign in the accumulation (the FD gradient is the source of truth, exactly as in Experiment A). If only some components match, STOP and report — that indicates a derivation error, not a sign flip.

- [ ] **Step 5: Commit**

```bash
git add baselines.py tests/test_baselines_mech.py
git commit -m "$(cat <<'EOF'
feat: implicit-adjoint gradient and GD baseline for the spring network

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Keystone — local rule ≡ true gradient as β→0

**Files:**
- Modify: `tests/test_mechanical_learning.py` (append)

**Interfaces:**
- Consumes: `free_phase_mech`, `nudged_phase_mech`, `local_update_mech` (mechanical_learning); `mech_adjoint_gradient` (baselines); `gradient_alignment` (baselines, from Experiment A).

This carries B's mechanism claim: the local two-phase `(Δk, Δℓ)` update converges to the negative true gradient as `β→0` (when no basin-jump occurs). Per A's review feedback, assert a concrete improvement margin, not just an endpoint inequality.

- [ ] **Step 1: Write the failing test (append to `tests/test_mechanical_learning.py`)**

```python
def test_local_update_aligns_with_true_gradient_as_beta_shrinks():
    import numpy as np
    from spring_network import SpringNetwork, triangulated_lattice, default_rest_lengths, dof
    from mechanical_learning import free_phase_mech, nudged_phase_mech, local_update_mech
    from baselines import mech_adjoint_gradient, gradient_alignment

    x0, edges = triangulated_lattice(3, 3)
    net = SpringNetwork(x0, edges, pinned_dofs=[dof(1, 0), dof(1, 1), dof(0, 0)],
                        input_dofs=[dof(0, 1), dof(2, 1), dof(6, 1)], output_dof=dof(8, 1))
    l = default_rest_lengths(net); k = np.linspace(0.8, 1.2, len(edges))
    pattern = np.array([1.0, -1.0, 1.0]); d = 0.3; target = 0.1

    gk, gl = mech_adjoint_gradient(net, k, l, np.array([pattern]), d, np.array([target]))
    neg_grad = -np.concatenate([gk, gl])

    rhos = []
    for beta in (0.2, 0.05, 0.01, 0.002):
        xf, _ = free_phase_mech(net, k, l, pattern, d)
        xn, _, _ = nudged_phase_mech(net, k, l, pattern, d, target, beta, xf)
        dk, dl = local_update_mech(net, k, l, xf, xn, beta, eta_k=1.0, eta_l=1.0)
        rhos.append(gradient_alignment(np.concatenate([dk, dl]), neg_grad)["rho"])

    assert rhos[-1] >= rhos[0] + 0.01            # alignment measurably improves as beta shrinks
    align = gradient_alignment(
        np.concatenate(local_update_mech(
            net, k, l,
            free_phase_mech(net, k, l, pattern, d)[0],
            nudged_phase_mech(net, k, l, pattern, d, target, 0.002,
                              free_phase_mech(net, k, l, pattern, d)[0])[0],
            0.002, eta_k=1.0, eta_l=1.0)),
        neg_grad)
    assert align["sign_match"] >= 0.9
    assert align["rho"] >= 0.95
```

- [ ] **Step 2: Run test to verify it fails (before Task 6 exists it would error; here it runs)**

Run: `python -m pytest tests/test_mechanical_learning.py::test_local_update_aligns_with_true_gradient_as_beta_shrinks -q`
Expected: it should PASS once Tasks 5 and 6 are correct. If it FAILS on `sign_match`/`rho` (anti-aligned or low), there is a sign/scaling mismatch between `local_update_mech` and `mech_adjoint_gradient` — reconcile by fixing `local_update_mech`'s sign (keep the FD-validated adjoint from Task 6 fixed). If it fails only the margin assertion `rhos[-1] >= rhos[0] + 0.01` (alignment already near-perfect at large β), widen the β range upward (e.g. start at 0.5) so the improvement is visible, or relax the margin with a logged note — do NOT weaken the small-β floor (`rho>=0.95`, `sign_match>=0.9`).

- [ ] **Step 3: No new implementation**

Behavior is produced by Task 5 (`local_update_mech`) and Task 6 (`mech_adjoint_gradient`). If failing, fix the sign in `local_update_mech` only.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mechanical_learning.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_mechanical_learning.py
git commit -m "$(cat <<'EOF'
test: keystone - mechanical local rule converges to true gradient as beta->0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: The experiment driver (`experiment_b.py`)

**Files:**
- Create: `experiment_b.py`
- Test: `tests/test_experiment_b.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run(config=None) -> dict` with keys `verdict` (a `verdict_b` dict), `seeds` (per-seed local/baseline separations + jump rates), `audit_fraction_physics_local`, `alignment`; `main()` prints a table and writes `experiment_b_results.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_b.py
from experiment_b import run


def test_run_emits_a_valid_verdict_and_is_physics_local():
    res = run({"rows": 3, "cols": 3, "epochs": 60, "baseline_epochs": 60,
               "beta": 0.02, "eta_k": 1.5, "eta_l": 1.5, "lr": 1.0, "seeds": [0, 1], "d": 0.3})
    assert res["verdict"]["status"] in {
        "MULTISTABILITY-FORECLOSE", "FORECLOSE-REPRESENTABILITY",
        "FORECLOSE-LEARNABILITY", "PASS", "AMBIGUOUS"}     # a VALID verdict, never asserting which
    assert res["audit_fraction_physics_local"] == 1.0     # headline training is honest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_experiment_b.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiment_b'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiment_b.py
"""Driver for Experiment B: train a 2D nonlinear spring network on parity-3 by the local mechanical
coupled-learning rule, compare to the offline adjoint-gradient baseline, measure separation/SNR,
gradient alignment, and the basin-jump rate, across seeds, and emit the five-way verdict. The
PASS/FORECLOSE outcome is genuinely unknown -- this driver computes it, it is not assumed."""
import json
import numpy as np
from spring_network import default_rest_lengths
from tasks import parity3_mech_task, two_sided_separation, verdict_b
from mechanical_learning import (free_phase_mech, nudged_phase_mech, local_update_mech,
                                 train_mech, readout)
from baselines import mech_adjoint_gradient, gd_baseline_mech, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "epochs": 400, "baseline_epochs": 400,
                  "beta": 0.01, "eta_k": 1.5, "eta_l": 1.5, "lr": 1.0,
                  "seeds": [0, 1, 2], "d": 0.3, "floor": 1e-3}


def _alignment(net, k, l, task):
    p = task["patterns"][0]; tgt = task["target_pos"]
    gk, gl = mech_adjoint_gradient(net, k, l, np.array([p]), task["d"], np.array([tgt]))
    xf, _ = free_phase_mech(net, k, l, p, task["d"])
    xn, _, _ = nudged_phase_mech(net, k, l, p, task["d"], tgt, 0.005, xf)
    dk, dl = local_update_mech(net, k, l, xf, xn, 0.005, eta_k=1.0, eta_l=1.0)
    return gradient_alignment(np.concatenate([dk, dl]), -np.concatenate([gk, gl]))


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    audit = AuditLog()
    seeds_out = []
    last = None
    for seed in cfg["seeds"]:
        task = parity3_mech_task(cfg["rows"], cfg["cols"], d=cfg["d"], seed=seed)
        net = task["net"]
        k0 = np.ones(len(net.edges)); l0 = default_rest_lengths(net)
        local = train_mech(task, k0, l0, cfg["beta"], cfg["eta_k"], cfg["eta_l"],
                           cfg["epochs"], audit=audit)
        base = gd_baseline_mech(task, k0, l0, cfg["baseline_epochs"], cfg["lr"])
        align = _alignment(net, local["k"], local["l"], task)
        seeds_out.append({"seed": seed,
                          "local_sep": local["sep_history"][-1],
                          "baseline_sep": base["sep"],
                          "jump_rate": local["jump_rate"],
                          "sign_match": align["sign_match"], "rho": align["rho"]})
        last = (local, base, align)
    # aggregate across seeds: robust PASS requires the WORST seed to clear the bar
    local_sep = min(s["local_sep"] for s in seeds_out)
    baseline_sep = min(s["baseline_sep"] for s in seeds_out)
    jump_rate = max(s["jump_rate"] for s in seeds_out)
    sign_match = min(s["sign_match"] for s in seeds_out)
    rho = min(s["rho"] for s in seeds_out)
    verdict = verdict_b(local_sep, baseline_sep, cfg["floor"], jump_rate, sign_match, rho)
    return {"verdict": verdict, "seeds": seeds_out,
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"],
            "alignment": {"sign_match": sign_match, "rho": rho}}


def main():
    res = run()
    print(f"{'seed':>5}{'local_sep':>12}{'base_sep':>12}{'jumps':>8}{'sign':>7}{'rho':>7}")
    for s in res["seeds"]:
        print(f"{s['seed']:>5}{s['local_sep']:>12.4e}{s['baseline_sep']:>12.4e}"
              f"{s['jump_rate']:>8.2f}{s['sign_match']:>7.2f}{s['rho']:>7.3f}")
    print(f"\nVERDICT: {res['verdict']['status']}  (SNR={res['verdict']['snr']:.2f}, "
          f"physics-local={res['audit_fraction_physics_local']:.2f})")
    with open("experiment_b_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote experiment_b_results.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_experiment_b.py -q`
Expected: PASS (1 passed). Then run the experiment itself:
Run: `python experiment_b.py`
Expected: a printed per-seed table and a `VERDICT: <one of the five statuses>` line, plus `experiment_b_results.json`. **Whatever the verdict is, that is the scientific result — do not tune toward PASS.**

- [ ] **Step 5: Commit**

```bash
git add experiment_b.py tests/test_experiment_b.py
git commit -m "$(cat <<'EOF'
feat: Experiment B driver (parity-3 verdict + JSON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass (A's 24 + B's new modules).

- [ ] **Run the experiment and record the result**

Run: `python experiment_b.py`
Record the per-seed table, the verdict, the basin-jump rate, and the alignment numbers.

- [ ] **Write the findings doc**

Write `docs/superpowers/specs/2026-06-19-experiment-b-findings.md` with an *Observed* section (per-seed separations, SNR, basin-jump rate, alignment, audit fraction, verdict) and a separate *Interpretation* section answering the SEED question: did signed/non-monotone elastic state clear the parity wall (PASS), fail to (FORECLOSE-representability → wall deeper than passivity → escalate to project 03), fail to learn it locally (FORECLOSE-learnability), or break the two-phase rule (MULTISTABILITY-FORECLOSE)? **Report the result honestly whatever it is — a clean FORECLOSE is a valid, valuable outcome.** Commit it.

---

## Self-review notes (author check against the spec)

- **Spec §2 architecture (parallel modules, A untouched, audit reuse):** Tasks 1–8; A's files only extended. ✓
- **Spec §3 spring model (energy/force/Hessian, smoothed-`m` asymmetry, triangulated rigidity, I/O encoding):** Tasks 1, 2, 4. ✓
- **Spec §3 solver (damped Newton + line search, warm-start, basin-jump):** Task 3. ✓
- **Spec §4 mechanical coupled-learning rule (both `k` and `ℓ`, edge-local, audit, clip):** Task 5. ✓
- **Spec §5 parity-3 task + two-sided separation + five-way verdict:** Task 4. ✓
- **Spec §6 implicit-adjoint gradient + FD cross-check + baseline + alignment:** Tasks 6, 7. ✓
- **Spec §7 testing (one module per source; keystone with a concrete margin; smoke asserts a valid verdict, never a specific one):** Tasks 1–8; Task 7 margin; Task 8 valid-verdict-only. ✓
- **Spec §8 verdicts / unknown outcome:** verdict_b (Task 4); findings doc records the result (Final verification). ✓
