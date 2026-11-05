# tests/test_experiment_b2.py
import numpy as np
from experiment_b2 import run, Probe, _init
from tasks import parity3_mech_task
from parity_analysis import parity_purity


def test_purity_loss_gradient_matches_finite_difference():
    task = parity3_mech_task(3, 3, d=0.1, seed=0)
    probe = Probe(task)
    theta = _init(task, 0)
    L, g = probe.purity_loss(theta)
    assert L == 1.0 - parity_purity(probe.readouts(*probe.split(theta))[1], probe.S)
    rng = np.random.default_rng(1); eps = 1e-6
    for i in rng.choice(len(theta), 4, replace=False):
        tp = theta.copy(); tp[i] += eps; tm = theta.copy(); tm[i] -= eps
        fd = (probe.purity_loss(tp)[0] - probe.purity_loss(tm)[0]) / (2 * eps)
        assert abs(g[i] - fd) < 1e-4 * max(1.0, abs(fd))


def test_run_emits_a_valid_verdict_and_is_all_offline():
    res = run({"rows": 3, "cols": 3, "ds": [0.1], "seeds": [0], "budgets": [3, 6],
               "de": {"d": 0.1, "seed": 0, "maxiter": 1, "popsize": 1,
                      "k_bounds": (0.1, 10.0), "l_rel_bounds": (0.8, 1.2)}})
    assert res["verdict"]["status"] in {"REPRESENTABLE", "FORECLOSE-REPRESENTABILITY-CONFIRMED", "AMBIGUOUS"}
    assert res["audit"]["all_offline"] is True          # a representability probe, never a learning run
    assert len(res["cells"]) == 3                        # purity + mse (L-BFGS) + DE
