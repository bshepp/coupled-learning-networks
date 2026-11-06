# tests/test_experiment_b3.py
import numpy as np
from experiment_b3 import run, validate
from tasks import parity3_mech_task
from spring_network import default_rest_lengths


def test_validate_counts_minima_and_measures_stability_on_the_untrained_lattice():
    task = parity3_mech_task(3, 3, d=0.1)
    net = task["net"]; k = np.ones(len(net.edges)); l = default_rest_lengths(net)
    val = validate(net, k, l, task["patterns"], 0.1, n_starts=3)
    assert all(v["n_minima"] == 1 and v["min_eig"] > 0 for v in val)      # small d: unique stable minimum


def test_run_emits_valid_per_d_verdicts_and_is_physics_local():
    res = run({"rows": 3, "cols": 3, "ds": [0.1], "seeds": [0], "epochs": 5, "n_starts": 2})
    assert res["verdict"]["0.1"]["status"] in {"PASS-ROBUST", "MULTISTABLE-HYSTERETIC",
                                               "FORECLOSE-REPRESENTABILITY", "AMBIGUOUS"}
    assert res["audit_fraction_physics_local"] == 1.0
