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
    targets = np.array([0.1, -0.1]); d = 0.1  # reduced from 0.3: alternating-sign patterns with
    # input nodes {0,2,6} become mechanically unstable (indefinite Hff) at d=0.3; d=0.1 is stable.
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
    assert res["sep"] > 0.05      # the offline optimizer genuinely separates the two classes


def test_mech_ceiling_beats_short_gd_on_allostery_and_is_offline():
    from tasks import allostery_mech_task
    from baselines import mech_ceiling
    from audit import AuditLog, OFFLINE_GRADIENT
    task = allostery_mech_task(3, 3, d_max=0.1, ratio=1.2, n_samples=4)   # ~3x the untrained ratio
    net = task["net"]
    k0 = np.ones(len(net.edges)); l0 = default_rest_lengths(net)
    audit = AuditLog()
    ceil = mech_ceiling(task, k0, l0, maxiter=200, audit=audit)
    gd = gd_baseline_mech(task, k0, l0, epochs=20, lr=0.5)
    assert ceil["final_error"] < 0.5 * gd["final_error"]           # a converged ceiling, not a budget artifact
    assert ceil["final_error"] < 0.1
    assert audit.summary()["counts"] == {OFFLINE_GRADIENT: audit.summary()["total"]}
