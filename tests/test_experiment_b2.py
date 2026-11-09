# tests/test_experiment_b2.py
import numpy as np
from experiment_b2 import run, Probe, _init
from tasks import parity3_mech_task
from parity_analysis import parity_cosine


def test_cosine_loss_gradient_matches_finite_difference_and_is_nonzero_at_init():
    task = parity3_mech_task(3, 3, d=0.1, seed=0)
    probe = Probe(task)
    theta = _init(task, 0)
    probe.set_sign(theta)
    L, g = probe.cosine_loss(theta)
    assert L == 1.0 - probe.sign * parity_cosine(probe.readouts(*probe.split(theta))[1], probe.S)
    assert np.linalg.norm(g) > 1e-3                     # the untrained lattice is NOT a stationary point
    rng = np.random.default_rng(1); eps = 1e-6
    for i in rng.choice(len(theta), 4, replace=False):
        tp = theta.copy(); tp[i] += eps; tm = theta.copy(); tm[i] -= eps
        fd = (probe.cosine_loss(tp)[0] - probe.cosine_loss(tm)[0]) / (2 * eps)
        assert abs(g[i] - fd) < 1e-4 * max(1.0, abs(fd))


def test_barrier_points_back_to_the_last_solvable_point():
    task = parity3_mech_task(3, 3, d=0.1, seed=0)
    probe = Probe(task)
    theta = _init(task, 0)
    L0, _ = probe.cosine_loss(theta)
    bad = theta.copy(); bad[probe.E:] *= 3.0

    class Unsolvable(RuntimeError):
        pass
    orig = probe.readouts
    probe.readouts = lambda k, l: (_ for _ in ()).throw(Unsolvable("forced genuine instability"))
    Lb, gb = probe.cosine_loss(bad)                   # the solve "fails": barrier back to theta
    probe.readouts = orig
    assert probe.n_fail == 1 and Lb > L0
    assert gb @ (theta - bad) < 0                     # descent direction leads back toward theta


def test_run_emits_a_valid_verdict_and_is_all_offline():
    res = run({"rows": 3, "cols": 3, "ds": [0.1], "seeds": [0], "budgets": [3, 6], "max_restarts": 1,
               "de": {"ds": [0.1], "seed": 0, "maxiter": 1, "popsize": 1,
                      "k_bounds": (0.2, 5.0), "l_rel_bounds": (0.8, 1.2)}})
    assert res["verdict"]["status"] in {"REPRESENTABLE", "FORECLOSE-REPRESENTABILITY-CONFIRMED", "AMBIGUOUS"}
    assert res["audit"]["all_offline"] is True          # a representability probe, never a learning run
    assert len(res["cells"]) == 3                        # cosine + mse (L-BFGS) + DE
    assert set(res["cells"][0]["purity_trace"]) == {"3", "6"}   # every checkpoint is filled
