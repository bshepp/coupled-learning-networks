# Coupled-Learning Networks (Experiment A + Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a substrate-agnostic coupled-learning harness and use it to show a passive resistor/flow network learns allostery + regression by a *local, physics-driven* rule whose updates match the true loss gradient.

**Architecture:** Approach 3 (hybrid reuse). Port `self-assembly/stage2_solver.py`'s batched Newton/Laplacian solve behind a thin `relax` seam. The new heart is the two-phase coupled-learning update (`coupled_learning.py`). Offline adjoint-gradient descent is the baseline; an audit hook logs the provenance of every weight change. Verdict = error-parity vs baseline **and** gradient alignment (sign-match/ρ) as β→0.

**Tech Stack:** Python 3, numpy, scipy, pytest. No jax/torch (the adjoint gradient is hand-written).

## Global Constraints

- **Stack:** numpy + scipy + pytest only. No deep-learning frameworks; the true gradient is hand-written (adjoint) and cross-checked against finite differences.
- **Layout:** flat modules at the repo root (matching the `self-assembly` house pattern); tests in `tests/` importing modules directly. Repo root is on `sys.path` for pytest (the repo-setup step adds a `conftest.py` at root so imports resolve).
- **House code pattern:** small focused modules; functions return numbers; each runnable module exposes a `main()` that prints a table and dumps JSON; one pytest module per source module.
- **Passivity (physical constraint):** trainable parameters are clipped to a positive range every step (`g_min > 0`). This is deliberate — it is *why* Experiment A cannot do parity. Defaults: `p_min=1e-3`, `p_max=10.0`.
- **Audit discipline:** every weight update is tagged `physics-local` (coupled learning) or `offline-gradient` (baseline). The headline coupled-learning run asserts 100% `physics-local`.
- **Reduction convention:** loss, task error, adjoint gradient, and the coupled-learning contrast are all **averaged over patterns** (full-batch mean) so the local update and the true gradient are directly comparable.
- **Commits:** every commit message ends with the trailer shown in Task 1, Step 5 (`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`). Apply it to all commits.
- **Repo prerequisite:** this plan assumes the repository skeleton (`.gitignore`, `requirements.txt`, `conftest.py`, `tests/`, `README.md`) already exists — created by the separate "repository setup" step that runs *after* this plan document and *before* execution.

---

## File structure

| File | Responsibility |
|---|---|
| `solver.py` | Batched Newton/Laplacian equilibrium solve + the `relax` seam (ported from `stage2_solver.py`) |
| `network.py` | Constitutive laws (resistor, Poiseuille); `Network` dataclass; `grid_network` builder; `conductance_edges` |
| `audit.py` | `AuditLog` + provenance constants |
| `tasks.py` | `Task` dataclass; `task_error`; `regression_task`; `allostery_task`; `verdict` |
| `coupled_learning.py` | `free_phase`, `nudged_phase`, `edge_drops`, `local_update`, `train` |
| `baselines.py` | `adjoint_gradient`, `gd_baseline`, `gradient_alignment` |
| `experiment_a.py` | `run(config)` driver + `main()` (table + JSON) |
| `tests/test_*.py` | one per source module |

---

## Task 1: The solver seam (`solver.py`)

**Files:**
- Create: `solver.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Produces: `assemble_laplacian(edges, V, weight_fn) -> (P,n,n)`; `solve_batched(edges, node_volts, floating, phi=_ones, dphi=_zeros, max_iter=60, tol=1e-11) -> (P,n)`; `relax(n_nodes, edges, clamp, phi=_ones, dphi=_zeros, **kw) -> (P,n_nodes)`. `edges` are `(a,b,g)` tuples (g = conductance weight). `clamp` is `{node_index: scalar_or_(P,)_array}`; floating nodes are those not in `clamp`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_solver.py
import numpy as np
import pytest
from solver import relax, assemble_laplacian


def test_equal_series_divider_midpoint():
    # node0=1.0 --g=1-- node1(float) --g=1-- node2=0.0  => V1 = 0.5
    V = relax(3, [(0, 1, 1.0), (1, 2, 1.0)], {0: 1.0, 2: 0.0})
    assert V.shape == (1, 3)
    assert V[0, 1] == pytest.approx(0.5, abs=1e-9)


def test_unequal_divider():
    # g01=2, g12=1 => V1 = g01/(g01+g12) = 2/3
    V = relax(3, [(0, 1, 2.0), (1, 2, 1.0)], {0: 1.0, 2: 0.0})
    assert V[0, 1] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_batched_patterns():
    V = relax(3, [(0, 1, 1.0), (1, 2, 1.0)], {0: np.array([1.0, 2.0]), 2: 0.0})
    assert V.shape == (2, 3)
    assert V[:, 1] == pytest.approx([0.5, 1.0], abs=1e-9)


def test_all_clamped_returns_clamps():
    # no floating nodes -> returns the clamp values unchanged
    V = relax(2, [(0, 1, 1.0)], {0: 1.0, 1: 0.25})
    assert V[0, 0] == pytest.approx(1.0)
    assert V[0, 1] == pytest.approx(0.25)


def test_laplacian_symmetric():
    V = np.zeros((1, 3))
    L = assemble_laplacian([(0, 1, 1.5), (1, 2, 0.7)], V, lambda u, w: np.full(u.shape, w))
    assert np.allclose(L[0], L[0].T)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_solver.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'solver'`.

- [ ] **Step 3: Write minimal implementation**

```python
# solver.py
"""Batched Newton/Laplacian equilibrium solve + the relax seam.

Ported and adapted from self-assembly/stage2_solver.py (batched Newton Kirchhoff
forward solve). Edges carry a conductance weight g; the actual edge conductance is
g*phi(dV) (phi==1 for the linear resistor/flow case used in Experiment A). relax()
is the substrate-agnostic seam: Experiment B's vector spring solver plugs in here.
"""
import numpy as np


def _ones(u):
    return np.ones_like(u)


def _zeros(u):
    return np.zeros_like(u)


def assemble_laplacian(edges, V, weight_fn):
    """Batched conductance Laplacian (P, n, n). weight_fn(u, w) -> (P,) conductance for
    edge weight w and per-pattern bias u = V_a - V_b."""
    P, n = V.shape
    L = np.zeros((P, n, n))
    for (a, b, w) in edges:
        u = V[:, a] - V[:, b]
        g = np.asarray(weight_fn(u, w), float)
        L[:, a, a] += g; L[:, b, b] += g
        L[:, a, b] -= g; L[:, b, a] -= g
    return L


def solve_batched(edges, node_volts, floating, phi=_ones, dphi=_zeros, max_iter=60, tol=1e-11):
    """Node voltages by Newton on the Kirchhoff residual with the analytic dynamic-Laplacian
    Jacobian, batched over all patterns. node_volts: (P, n) with np.nan at floating nodes."""
    V = np.array(node_volts, float)
    P, n = V.shape
    fset = set(int(i) for i in floating)
    fidx = np.array(sorted(fset), int)
    fixed = np.array([i for i in range(n) if i not in fset], int)
    if fidx.size == 0:                      # nothing floats -> clamps are the answer
        return V
    V[:, fidx] = 0.0

    L0 = assemble_laplacian(edges, V, lambda u, w: np.full(u.shape, float(w)))
    Lff0 = L0[:, fidx[:, None], fidx[None, :]]
    Lfb0 = L0[:, fidx[:, None], fixed[None, :]]
    rhs0 = -np.einsum('pij,pj->pi', Lfb0, V[:, fixed])
    V[:, fidx] = np.linalg.solve(Lff0, rhs0[..., None])[..., 0]

    g_fn = lambda u, w: float(w) * np.asarray(phi(u), float)
    gdyn_fn = lambda u, w: float(w) * (np.asarray(phi(u), float) + u * np.asarray(dphi(u), float))

    last = np.inf
    for _ in range(max_iter):
        Lg = assemble_laplacian(edges, V, g_fn)
        R = np.einsum('pij,pj->pi', Lg, V)[:, fidx]
        last = float(np.max(np.abs(R)))
        if last < tol:
            return V
        Jd = assemble_laplacian(edges, V, gdyn_fn)
        Jff = Jd[:, fidx[:, None], fidx[None, :]]
        dVf = np.linalg.solve(Jff, (-R)[..., None])[..., 0]
        V[:, fidx] += dVf
    raise RuntimeError(f"solve_batched did not converge in {max_iter} iters (max|residual|={last:.2e})")


def relax(n_nodes, edges, clamp, phi=_ones, dphi=_zeros, **kw):
    """Solve node voltages. clamp: {node_index: scalar or (P,) array}. Floating nodes are
    those not present in clamp. Returns (P, n_nodes)."""
    P = 1
    for v in clamp.values():
        P = max(P, np.atleast_1d(v).shape[0])
    node_volts = np.full((P, n_nodes), np.nan)
    for node, val in clamp.items():
        node_volts[:, node] = val
    floating = [i for i in range(n_nodes) if i not in clamp]
    return solve_batched(edges, node_volts, floating, phi, dphi, **kw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_solver.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add solver.py tests/test_solver.py
git commit -m "$(cat <<'EOF'
feat: solver seam (ported batched Newton/Laplacian relax)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Network model & constitutive laws (`network.py`)

**Files:**
- Create: `network.py`
- Test: `tests/test_network.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Law(name, g_of_p, dg_dp)`; `RESISTOR` (a `Law`); `poiseuille_law(C=1.0) -> Law`; `Network(n_nodes, edges, inputs, outputs, ground)` (dataclass; `edges` = list of `(a,b)`); `grid_network(rows, cols, n_in=1, n_out=1) -> Network`; `conductance_edges(net, g) -> [(a,b,float)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_network.py
import numpy as np
import pytest
from network import RESISTOR, poiseuille_law, Network, grid_network, conductance_edges


def test_resistor_law_identity():
    p = np.array([0.5, 2.0])
    assert RESISTOR.g_of_p(p) == pytest.approx(p)
    assert RESISTOR.dg_dp(p) == pytest.approx([1.0, 1.0])


def test_poiseuille_law_and_derivative():
    law = poiseuille_law(C=2.0)
    p = np.array([1.0, 1.5])
    assert law.g_of_p(p) == pytest.approx(2.0 * p ** 4)
    # finite-difference check of dg_dp
    eps = 1e-6
    fd = (law.g_of_p(p + eps) - law.g_of_p(p - eps)) / (2 * eps)
    assert law.dg_dp(p) == pytest.approx(fd, rel=1e-4)


def test_grid_network_topology():
    net = grid_network(3, 3, n_in=2, n_out=1)
    assert net.n_nodes == 9
    assert len(net.edges) == 12          # 3*2 horizontal + 2*3 vertical
    assert len(net.inputs) == 2 and len(net.outputs) == 1
    # inputs/outputs/ground all distinct
    special = net.inputs + net.outputs + [net.ground]
    assert len(set(special)) == len(special)


def test_conductance_edges_shapes():
    net = grid_network(2, 2)
    g = np.full(len(net.edges), 0.7)
    ce = conductance_edges(net, g)
    assert len(ce) == len(net.edges)
    a, b, w = ce[0]
    assert (a, b) == net.edges[0] and w == pytest.approx(0.7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_network.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'network'`.

- [ ] **Step 3: Write minimal implementation**

```python
# network.py
"""Substrate description for coupled-learning networks: pluggable constitutive laws,
a Network dataclass, and builders. The trainable per-edge parameter p maps to a
conductance g via the constitutive law (resistor: g=p; Poiseuille: g=C*p**4)."""
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Law:
    name: str
    g_of_p: Callable      # conductance from trainable parameter
    dg_dp: Callable       # d(conductance)/d(parameter), for the coupled-learning chain rule


RESISTOR = Law(
    "resistor",
    lambda p: np.asarray(p, float),
    lambda p: np.ones_like(np.asarray(p, float)),
)


def poiseuille_law(C=1.0):
    """Hagen-Poiseuille: parameter p = radius, conductance g = C * r**4."""
    return Law(
        "poiseuille",
        lambda p: C * np.asarray(p, float) ** 4,
        lambda p: 4.0 * C * np.asarray(p, float) ** 3,
    )


@dataclass
class Network:
    n_nodes: int
    edges: List[tuple]            # list of (a, b)
    inputs: List[int]
    outputs: List[int]
    ground: Optional[int] = None


def grid_network(rows, cols, n_in=1, n_out=1):
    """2-D grid graph. Inputs = top n_in of the left column; outputs = bottom n_out of the
    right column; ground = bottom-left node. Requires n_in <= rows, n_out <= rows."""
    idx = lambda r, c: r * cols + c
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    inputs = [idx(r, 0) for r in range(n_in)]
    outputs = [idx(rows - 1 - j, cols - 1) for j in range(n_out)]
    ground = idx(rows - 1, 0)
    return Network(rows * cols, edges, inputs, outputs, ground)


def conductance_edges(net, g):
    """Solver edges [(a, b, g_e)] from a Network and per-edge conductances g (len = #edges)."""
    return [(a, b, float(ge)) for (a, b), ge in zip(net.edges, np.asarray(g, float))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_network.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add network.py tests/test_network.py
git commit -m "$(cat <<'EOF'
feat: network model and pluggable constitutive laws

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: The audit hook (`audit.py`)

**Files:**
- Create: `audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces: constants `PHYSICS_LOCAL`, `OFFLINE_GRADIENT`; `AuditLog` with `.record(provenance, n=1)`, `.summary() -> dict`, `.assert_physics_local() -> True | raises`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit.py
import pytest
from audit import AuditLog, PHYSICS_LOCAL, OFFLINE_GRADIENT


def test_all_physics_local():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 12)
    log.record(PHYSICS_LOCAL, 12)
    s = log.summary()
    assert s["fraction_physics_local"] == pytest.approx(1.0)
    assert log.assert_physics_local() is True


def test_offline_gradient_flagged():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 5)
    log.record(OFFLINE_GRADIENT, 5)
    s = log.summary()
    assert s["fraction_physics_local"] == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        log.assert_physics_local()


def test_summary_counts():
    log = AuditLog()
    log.record(PHYSICS_LOCAL, 3)
    log.record(OFFLINE_GRADIENT, 7)
    s = log.summary()
    assert s["counts"][PHYSICS_LOCAL] == 3
    assert s["counts"][OFFLINE_GRADIENT] == 7
    assert s["total"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'audit'`.

- [ ] **Step 3: Write minimal implementation**

```python
# audit.py
"""The audit hook: makes "who computes the update?" a logged, checkable fact.
Every weight change is tagged at its source as physics-local or offline-gradient."""

PHYSICS_LOCAL = "physics-local"
OFFLINE_GRADIENT = "offline-gradient"


class AuditLog:
    def __init__(self):
        self.records = []

    def record(self, provenance, n=1):
        self.records.append((provenance, int(n)))

    def summary(self):
        counts = {}
        for prov, n in self.records:
            counts[prov] = counts.get(prov, 0) + n
        total = sum(counts.values())
        frac = counts.get(PHYSICS_LOCAL, 0) / total if total else 0.0
        return {"counts": counts, "total": total, "fraction_physics_local": frac}

    def assert_physics_local(self):
        s = self.summary()
        if s["counts"].get(OFFLINE_GRADIENT, 0) > 0:
            raise AssertionError(f"headline run used offline-gradient updates: {s}")
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audit.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add audit.py tests/test_audit.py
git commit -m "$(cat <<'EOF'
feat: audit hook for update provenance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Tasks, error metric & verdict (`tasks.py`)

**Files:**
- Create: `tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `grid_network`, `Network` from `network.py`.
- Produces: `Task(name, net, inputs, targets)` (dataclass; `inputs` shape `(P, n_in)`, `targets` shape `(P, n_out)`); `task_error(outputs, targets) -> float` (normalized MSE); `regression_task(...) -> Task`; `allostery_task(...) -> Task`; `verdict(local_error, baseline_error, sign_match, rho, tol=0.15, sm_thresh=0.95, rho_thresh=0.9) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks.py
import numpy as np
import pytest
from tasks import task_error, regression_task, allostery_task, verdict


def test_task_error_zero_and_positive():
    t = np.array([[0.4], [0.6]])
    assert task_error(t, t) == pytest.approx(0.0)
    assert task_error(t + 0.1, t) > 0.0


def test_regression_targets_representable():
    # passive network output obeys the maximum principle: target must lie within input range
    task = regression_task()
    assert task.targets.min() >= 0.0
    assert task.targets.max() <= task.inputs.max() + 1e-9


def test_allostery_targets_are_ratio_of_source():
    task = allostery_task(ratio=0.5)
    assert task.targets == pytest.approx(0.5 * task.inputs)


def test_verdict_branches():
    assert verdict(0.01, 0.01, 1.0, 1.0)["status"] == "PASS"
    assert verdict(0.5, 0.01, 1.0, 1.0)["status"] == "FORECLOSE"
    assert verdict(0.01, 0.01, 0.5, 0.5)["status"] == "AMBIGUOUS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tasks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tasks.py
"""Friendly, representable tasks for Experiment A (allostery + regression), the
normalized error metric, and the pre-registered verdict logic. Targets are chosen
inside the passive network's achievable range (the maximum principle: output is a
sub-convex combination of clamped inputs and ground)."""
import numpy as np
from dataclasses import dataclass
from network import grid_network, Network


@dataclass
class Task:
    name: str
    net: Network
    inputs: np.ndarray     # (P, n_in)
    targets: np.ndarray    # (P, n_out)


def task_error(outputs, targets):
    """Normalized MSE: mean((out - tgt)^2) / (mean(tgt^2) + eps)."""
    outputs = np.asarray(outputs, float)
    targets = np.asarray(targets, float)
    return float(np.mean((outputs - targets) ** 2) / (np.mean(targets ** 2) + 1e-12))


def regression_task(rows=3, cols=3, n_in=2, n_out=1, n_samples=8, weights=(0.4, 0.3), seed=0):
    """Map input-node voltages to an output voltage via a positive linear map whose
    weights sum to < 1 (so targets stay inside the achievable range)."""
    net = grid_network(rows, cols, n_in=n_in, n_out=n_out)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.2, 1.0, size=(n_samples, n_in))
    w = np.asarray(weights, float)[:n_in]
    Y = (X @ w).reshape(-1, n_out)
    return Task("regression", net, X, Y)


def allostery_task(rows=3, cols=3, ratio=0.5, n_samples=6, seed=1):
    """Train so a single source drive produces target = ratio * source at a distant node."""
    net = grid_network(rows, cols, n_in=1, n_out=1)
    rng = np.random.default_rng(seed)
    V_src = rng.uniform(0.3, 1.0, size=(n_samples, 1))
    Y = ratio * V_src
    return Task("allostery", net, V_src, Y)


def verdict(local_error, baseline_error, sign_match, rho, tol=0.15, sm_thresh=0.95, rho_thresh=0.9):
    """PASS: local rule reaches the offline baseline AND its update aligns with the true
    gradient. FORECLOSE: local rule cannot reach the baseline. AMBIGUOUS: otherwise."""
    learns = local_error <= baseline_error * (1.0 + tol)
    aligned = (sign_match >= sm_thresh) and (rho >= rho_thresh)
    if learns and aligned:
        status = "PASS"
    elif not learns:
        status = "FORECLOSE"
    else:
        status = "AMBIGUOUS"
    return {"status": status, "local_error": float(local_error),
            "baseline_error": float(baseline_error), "sign_match": float(sign_match),
            "rho": float(rho)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tasks.py tests/test_tasks.py
git commit -m "$(cat <<'EOF'
feat: tasks, error metric, and pre-registered verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: The coupled-learning rule (`coupled_learning.py`)

**Files:**
- Create: `coupled_learning.py`
- Test: `tests/test_coupled_learning.py`

**Interfaces:**
- Consumes: `relax` (solver); `conductance_edges`, `Network`, `Law` (network); `task_error` (tasks); `PHYSICS_LOCAL`, `AuditLog` (audit).
- Produces: `free_phase(net, g, inputs) -> (P,n)`; `nudged_phase(net, g, inputs, targets, beta, free_out=None) -> (P,n)`; `edge_drops(net, V) -> (P,E)`; `local_update(net, params, law, V_free, V_nudged, beta, eta, audit=None) -> (E,)`; `train(net, params0, law, task, beta, eta, epochs, p_min=1e-3, p_max=10.0, audit=None) -> dict` with keys `params`, `history`, `final_error`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coupled_learning.py
import numpy as np
import pytest
from network import grid_network, RESISTOR
from tasks import regression_task
from coupled_learning import free_phase, nudged_phase, train
from audit import AuditLog


def test_nudge_moves_output_toward_target():
    net = grid_network(3, 3, n_in=1, n_out=1)
    g = np.full(len(net.edges), 1.0)
    inputs = np.array([[1.0]])
    targets = np.array([[0.9]])     # above the free output of a uniform net
    Vf = free_phase(net, g, inputs)
    free_out = Vf[:, net.outputs]
    Vn = nudged_phase(net, g, inputs, targets, beta=0.1, free_out=free_out)
    nud_out = Vn[:, net.outputs]
    # nudged output sits strictly between free output and target
    assert free_out[0, 0] < nud_out[0, 0] <= targets[0, 0] + 1e-9


def test_training_reduces_error():
    task = regression_task(seed=0)
    params0 = np.full(len(task.net.edges), 1.0)
    audit = AuditLog()
    res = train(task.net, params0, RESISTOR, task, beta=0.05, eta=0.5, epochs=200, audit=audit)
    assert res["history"][-1] < res["history"][0]
    assert res["final_error"] < 0.1
    assert audit.assert_physics_local() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coupled_learning.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'coupled_learning'`.

- [ ] **Step 3: Write minimal implementation**

```python
# coupled_learning.py
"""The heart: the two-phase coupled-learning rule (Stern, Hexner, Rocks, Liu, PRX 2021).
Free phase clamps inputs; nudged phase additionally clamps outputs a small step toward
the target; each edge updates from the LOCAL power contrast between the two relaxed states.
No global error signal touches a weight -- that is the audit point made concrete."""
import numpy as np
from solver import relax
from network import conductance_edges
from tasks import task_error
from audit import PHYSICS_LOCAL


def _input_clamp(net, inputs):
    P = inputs.shape[0]
    clamp = {node: inputs[:, k] for k, node in enumerate(net.inputs)}
    if net.ground is not None:
        clamp[net.ground] = np.zeros(P)
    return clamp


def free_phase(net, g, inputs):
    edges = conductance_edges(net, g)
    return relax(net.n_nodes, edges, _input_clamp(net, inputs))


def nudged_phase(net, g, inputs, targets, beta, free_out=None):
    if free_out is None:
        free_out = free_phase(net, g, inputs)[:, net.outputs]
    clamp = _input_clamp(net, inputs)
    for j, onode in enumerate(net.outputs):
        clamp[onode] = free_out[:, j] + beta * (targets[:, j] - free_out[:, j])
    edges = conductance_edges(net, g)
    return relax(net.n_nodes, edges, clamp)


def edge_drops(net, V):
    """Voltage drop across each edge for each pattern: (P, E)."""
    e = np.array(net.edges, int)
    return V[:, e[:, 0]] - V[:, e[:, 1]]


def local_update(net, params, law, V_free, V_nudged, beta, eta, audit=None):
    """Δp = (eta/beta) * dg/dp * mean_patterns[(ΔV_free)^2 - (ΔV_nudged)^2].

    Sign: nudging the output toward the target shrinks the drop on edges feeding it, so
    the contrast is positive there and raises their conductance -- gradient descent on the
    task loss. This sign is enforced by the keystone alignment test in Task 7."""
    dvf = edge_drops(net, V_free)
    dvn = edge_drops(net, V_nudged)
    contrast = np.mean(dvf ** 2 - dvn ** 2, axis=0)          # (E,)
    dp = (eta / beta) * law.dg_dp(np.asarray(params, float)) * contrast
    if audit is not None:
        audit.record(PHYSICS_LOCAL, len(np.atleast_1d(params)))
    return dp


def train(net, params0, law, task, beta, eta, epochs, p_min=1e-3, p_max=10.0, audit=None):
    params = np.array(params0, float)
    history = []
    for _ in range(epochs):
        g = law.g_of_p(params)
        Vf = free_phase(net, g, task.inputs)
        out = Vf[:, net.outputs]
        history.append(task_error(out, task.targets))
        Vn = nudged_phase(net, g, task.inputs, task.targets, beta, free_out=out)
        dp = local_update(net, params, law, Vf, Vn, beta, eta, audit=audit)
        params = np.clip(params + dp, p_min, p_max)
    g = law.g_of_p(params)
    out = free_phase(net, g, task.inputs)[:, net.outputs]
    return {"params": params, "history": history, "final_error": task_error(out, task.targets)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coupled_learning.py -q`
Expected: PASS (2 passed). If `test_training_reduces_error` does not reach `< 0.1`, raise `epochs` or `eta` (do not change the rule).

- [ ] **Step 5: Commit**

```bash
git add coupled_learning.py tests/test_coupled_learning.py
git commit -m "$(cat <<'EOF'
feat: two-phase coupled-learning rule (the heart)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Offline baseline & gradient alignment (`baselines.py`)

**Files:**
- Create: `baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `free_phase`, `edge_drops` (coupled_learning); `conductance_edges`, `solver.relax` (network/solver); `task_error` (tasks); `OFFLINE_GRADIENT` (audit).
- Produces: `adjoint_gradient(net, g, task) -> (E,)` (mean over patterns, dL/dg with `L = 0.5*mean_p Σ_out (V_o - t_o)^2`); `gd_baseline(net, params0, law, task, epochs, lr, p_min=1e-3, p_max=10.0, audit=None) -> dict` with keys `params`, `history`, `final_error`; `gradient_alignment(delta_local, neg_grad) -> dict` with keys `sign_match`, `rho`, `cosine`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pytest
from network import grid_network, RESISTOR
from tasks import regression_task
from baselines import adjoint_gradient, gd_baseline, gradient_alignment


def _fd_gradient(net, g, task):
    from coupled_learning import free_phase
    eps = 1e-6
    grad = np.zeros(len(net.edges))
    for e in range(len(net.edges)):
        gp = g.copy(); gp[e] += eps
        gm = g.copy(); gm[e] -= eps
        op = free_phase(net, gp, task.inputs)[:, net.outputs]
        om = free_phase(net, gm, task.inputs)[:, net.outputs]
        lp = 0.5 * np.mean(np.sum((op - task.targets) ** 2, axis=1))
        lm = 0.5 * np.mean(np.sum((om - task.targets) ** 2, axis=1))
        grad[e] = (lp - lm) / (2 * eps)
    return grad


def test_adjoint_matches_finite_difference():
    task = regression_task(seed=2)
    g = np.linspace(0.5, 1.5, len(task.net.edges))
    assert adjoint_gradient(task.net, g, task) == pytest.approx(_fd_gradient(task.net, g, task), abs=1e-5)


def test_gd_baseline_converges():
    task = regression_task(seed=0)
    params0 = np.full(len(task.net.edges), 1.0)
    res = gd_baseline(task.net, params0, RESISTOR, task, epochs=300, lr=0.5)
    assert res["final_error"] < 0.05


def test_gradient_alignment_extremes():
    v = np.array([1.0, -2.0, 3.0, -0.5])
    same = gradient_alignment(v, v)
    assert same["sign_match"] == pytest.approx(1.0)
    assert same["rho"] == pytest.approx(1.0)
    assert same["cosine"] == pytest.approx(1.0)
    opp = gradient_alignment(v, -v)
    assert opp["sign_match"] == pytest.approx(0.0)
    assert opp["cosine"] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'baselines'`.

- [ ] **Step 3: Write minimal implementation**

```python
# baselines.py
"""The offline 'CPU computes the update' control: the exact loss gradient by the adjoint
method (validated against finite differences) and gradient descent on it. Plus the
gradient-alignment statistics that test whether the local rule tracks the true gradient.
Loss L = 0.5 * mean_patterns( sum_outputs (V_o - t_o)^2 )."""
import numpy as np
from network import conductance_edges
from coupled_learning import free_phase, edge_drops


def _full_laplacian(net, g):
    n = net.n_nodes
    L = np.zeros((n, n))
    for (a, b), ge in zip(net.edges, g):
        L[a, a] += ge; L[b, b] += ge
        L[a, b] -= ge; L[b, a] -= ge
    return L


def adjoint_gradient(net, g, task):
    """dL/dg_e by the adjoint method, averaged over patterns. For each pattern: solve the
    forward floating-node system, then the adjoint system (same symmetric L_ff), then
    dL/dg_e = (lambda_i - lambda_j)(V_i - V_j) with lambda=0 at clamped nodes."""
    g = np.asarray(g, float)
    n = net.n_nodes
    clamped = set(net.inputs) | ({net.ground} if net.ground is not None else set())
    floating = [i for i in range(n) if i not in clamped]
    fpos = {node: k for k, node in enumerate(floating)}
    L = _full_laplacian(net, g)
    Lff = L[np.ix_(floating, floating)]
    P = task.inputs.shape[0]
    grad = np.zeros(len(net.edges))

    # forward node voltages for all patterns (reuse the network solver)
    V = free_phase(net, g, task.inputs)                       # (P, n)
    out_idx = list(net.outputs)

    for p in range(P):
        r = np.zeros(len(floating))                           # dL/dV restricted to floating
        for j, onode in enumerate(out_idx):
            r[fpos[onode]] = V[p, onode] - task.targets[p, j]
        lam_f = np.linalg.solve(Lff, -r)                      # L_ff^T lam = -dL/dV_f
        lam = np.zeros(n)
        for node, k in fpos.items():
            lam[node] = lam_f[k]
        for e, (a, b) in enumerate(net.edges):
            grad[e] += (lam[a] - lam[b]) * (V[p, a] - V[p, b])
    return grad / P


def gd_baseline(net, params0, law, task, epochs, lr, p_min=1e-3, p_max=10.0, audit=None):
    from audit import OFFLINE_GRADIENT
    from tasks import task_error
    params = np.array(params0, float)
    history = []
    for _ in range(epochs):
        g = law.g_of_p(params)
        out = free_phase(net, g, task.inputs)[:, net.outputs]
        history.append(task_error(out, task.targets))
        grad_g = adjoint_gradient(net, g, task)
        grad_p = grad_g * law.dg_dp(params)
        params = np.clip(params - lr * grad_p, p_min, p_max)
        if audit is not None:
            audit.record(OFFLINE_GRADIENT, len(np.atleast_1d(params)))
    g = law.g_of_p(params)
    out = free_phase(net, g, task.inputs)[:, net.outputs]
    return {"params": params, "history": history, "final_error": task_error(out, task.targets)}


def gradient_alignment(delta_local, neg_grad):
    a = np.asarray(delta_local, float)
    b = np.asarray(neg_grad, float)
    sign_match = float(np.mean(np.sign(a) == np.sign(b)))
    rho = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"sign_match": sign_match, "rho": rho, "cosine": cosine}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines.py -q`
Expected: PASS (3 passed). If `test_adjoint_matches_finite_difference` fails by a global sign, flip the sign in the `grad[e] +=` accumulation and re-run — the finite-difference test is the source of truth for the sign.

- [ ] **Step 5: Commit**

```bash
git add baselines.py tests/test_baselines.py
git commit -m "$(cat <<'EOF'
feat: adjoint-gradient baseline and alignment stats

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Keystone — local rule ≡ true gradient as β→0

**Files:**
- Modify: `tests/test_coupled_learning.py` (append)

**Interfaces:**
- Consumes: `local_update`, `free_phase`, `nudged_phase` (coupled_learning); `adjoint_gradient`, `gradient_alignment` (baselines).

This is the test that carries the whole thesis: the local two-phase update must converge to the negative true gradient as the nudge `β→0`.

- [ ] **Step 1: Write the failing test (append to `tests/test_coupled_learning.py`)**

```python
def test_local_update_aligns_with_true_gradient_as_beta_shrinks():
    import numpy as np
    from network import grid_network, RESISTOR
    from tasks import regression_task
    from coupled_learning import free_phase, nudged_phase, local_update
    from baselines import adjoint_gradient, gradient_alignment

    task = regression_task(seed=3)
    net = task.net
    g = np.linspace(0.6, 1.4, len(net.edges))
    neg_grad = -adjoint_gradient(net, g, task)

    rhos = []
    for beta in (0.2, 0.05, 0.01, 0.002):
        Vf = free_phase(net, g, task.inputs)
        out = Vf[:, net.outputs]
        Vn = nudged_phase(net, g, task.inputs, task.targets, beta, free_out=out)
        dp = local_update(net, g, RESISTOR, Vf, Vn, beta, eta=1.0)
        rhos.append(gradient_alignment(dp, neg_grad)["rho"])

    # alignment improves as beta shrinks, and is near-perfect in the small-beta limit
    assert rhos[-1] > rhos[0]
    align = gradient_alignment(
        local_update(
            net, g, RESISTOR,
            free_phase(net, g, task.inputs),
            nudged_phase(net, g, task.inputs, task.targets, 0.002,
                         free_out=free_phase(net, g, task.inputs)[:, net.outputs]),
            0.002, eta=1.0),
        neg_grad)
    assert align["sign_match"] >= 0.95
    assert align["rho"] >= 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coupled_learning.py::test_local_update_aligns_with_true_gradient_as_beta_shrinks -q`
Expected: FAIL (until the alignment holds — if it fails on `sign_match` the `local_update` sign is wrong; reconcile with Task 6's adjoint sign).

- [ ] **Step 3: No new implementation needed**

The behavior is already produced by `local_update` (Task 5) and `adjoint_gradient` (Task 6). If the test fails, the bug is a sign mismatch between the two — fix `local_update`'s sign so it aligns with `-adjoint_gradient` (gradient descent), keeping Task 6's finite-difference-validated adjoint fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coupled_learning.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_coupled_learning.py
git commit -m "$(cat <<'EOF'
test: keystone - local rule converges to true gradient as beta->0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: The experiment driver (`experiment_a.py`)

**Files:**
- Create: `experiment_a.py`
- Test: `tests/test_experiment_a.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run(config=None) -> dict` mapping each task name to `{local_error, baseline_error, sign_match, rho, verdict, audit_fraction_physics_local}`, plus a top-level `audit` summary; `main()` prints a table and writes `experiment_a_results.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_a.py
from experiment_a import run


def test_run_smoke_passes_and_is_physics_local():
    res = run({"epochs": 200, "baseline_epochs": 300, "beta": 0.01, "eta": 0.5, "lr": 0.5})
    for name in ("allostery", "regression"):
        assert name in res
        assert res[name]["audit_fraction_physics_local"] == 1.0   # headline run is honest
        assert res[name]["verdict"]["status"] in ("PASS", "AMBIGUOUS", "FORECLOSE")
    # on these friendly tasks we expect the thesis to hold
    assert res["regression"]["verdict"]["status"] == "PASS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_experiment_a.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiment_a'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiment_a.py
"""Driver for Experiment A: train the resistor network by the local coupled-learning rule
on allostery + regression, compare to the offline adjoint-gradient baseline, measure
gradient alignment, and emit the pre-registered verdict. Answers 'who computes the update?'
with a logged, asserted audit trail."""
import json
import numpy as np
from network import RESISTOR
from tasks import regression_task, allostery_task, verdict
from coupled_learning import train, free_phase, nudged_phase, local_update
from baselines import gd_baseline, adjoint_gradient, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"epochs": 400, "baseline_epochs": 400, "beta": 0.01, "eta": 0.5, "lr": 0.5}


def _one_task(task, cfg):
    net = task.net
    params0 = np.full(len(net.edges), 1.0)

    audit = AuditLog()
    local = train(net, params0, RESISTOR, task, cfg["beta"], cfg["eta"], cfg["epochs"], audit=audit)

    base = gd_baseline(net, params0, RESISTOR, task, cfg["baseline_epochs"], cfg["lr"])

    g = RESISTOR.g_of_p(local["params"])
    Vf = free_phase(net, g, task.inputs)
    Vn = nudged_phase(net, g, task.inputs, task.targets, cfg["beta"], free_out=Vf[:, net.outputs])
    dp = local_update(net, g, RESISTOR, Vf, Vn, cfg["beta"], eta=1.0)
    align = gradient_alignment(dp, -adjoint_gradient(net, g, task))

    v = verdict(local["final_error"], base["final_error"], align["sign_match"], align["rho"])
    return {"local_error": local["final_error"], "baseline_error": base["final_error"],
            "sign_match": align["sign_match"], "rho": align["rho"], "verdict": v,
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"]}


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    out = {}
    out["allostery"] = _one_task(allostery_task(), cfg)
    out["regression"] = _one_task(regression_task(), cfg)
    return out


def main():
    res = run()
    print(f"{'task':<12}{'local_err':>12}{'base_err':>12}{'sign':>8}{'rho':>8}{'verdict':>12}")
    for name, r in res.items():
        print(f"{name:<12}{r['local_error']:>12.4e}{r['baseline_error']:>12.4e}"
              f"{r['sign_match']:>8.2f}{r['rho']:>8.3f}{r['verdict']['status']:>12}")
    with open("experiment_a_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote experiment_a_results.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_experiment_a.py -q`
Expected: PASS (1 passed). Then run the experiment itself:
Run: `python experiment_a.py`
Expected: a printed table with `regression` showing `verdict = PASS`, plus `experiment_a_results.json`.

- [ ] **Step 5: Commit**

```bash
git add experiment_a.py tests/test_experiment_a.py
git commit -m "$(cat <<'EOF'
feat: Experiment A driver (verdict + JSON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass (test_solver, test_network, test_audit, test_tasks, test_coupled_learning ×3, test_baselines, test_experiment_a).

- [ ] **Reproduce the headline result**

Run: `python experiment_a.py`
Expected: `regression` PASS; `allostery` PASS or (acceptably) AMBIGUOUS; `audit_fraction_physics_local == 1.0` for both — the physics computed the update.

- [ ] **Record findings**

Write `docs/superpowers/specs/2026-06-18-experiment-a-findings.md` with an *Observed* section (the verdict table, the β-sweep alignment numbers, the audit fraction) and a separate *Interpretation* section (does the dual-role learning thesis hold in a resistor substrate?). Commit it.

---

## Self-review notes (author check against the spec)

- **Spec §2 architecture (Approach 3, port-not-import):** Task 1 ports the solver behind `relax`. ✓
- **Spec §3 coupled-learning rule + β-sweep + sign-via-test:** Tasks 5 and 7. ✓
- **Spec §4 tasks (allostery + regression) + passivity clip:** Task 4 + `train`/`gd_baseline` clipping. ✓
- **Spec §5 baselines (adjoint, FD cross-check) + gradient alignment + verdict:** Tasks 4, 6. ✓
- **Spec §6 audit hook (100% physics-local assertion):** Task 3 + asserted in Tasks 5, 8. ✓
- **Spec §7 testing (one module per source module, keystone test):** Tasks 1–8. ✓
- **Deferred (B, nonlinear conductance, hardware):** not in this plan, by design. ✓
- **Repo setup (spec §8):** handled by the separate post-plan step, not a code task. ✓
