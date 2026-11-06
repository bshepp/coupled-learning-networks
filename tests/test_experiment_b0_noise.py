# tests/test_experiment_b0_noise.py
import numpy as np
from experiment_b0_noise import run, contrast_scale
from tasks import allostery_mech_task
from spring_network import default_rest_lengths


def test_contrast_scale_shrinks_with_beta():
    task = allostery_mech_task(3, 3, d_max=0.1, ratio=1.0, n_samples=3)
    net = task["net"]; k = np.ones(len(net.edges)); l = default_rest_lengths(net)
    med_small, _ = contrast_scale(task, k, l, 0.01)
    med_large, _ = contrast_scale(task, k, l, 0.1)
    assert 0 < med_small < med_large          # the signal the read noise competes with is ~ beta


def test_run_emits_a_valid_verdict_and_stays_physics_local():
    res = run({"rows": 3, "cols": 3, "n_samples": 3, "sigmas": [0.0, 1e-2], "betas": [0.05],
               "epochs": 8, "seeds": [0]})
    assert res["verdict"]["status"] in {"UNTESTED", "H5-MECHANISM-CONFIRMED", "H5-MECHANISM-REFUTED", "AMBIGUOUS"}
    assert res["audit_fraction_physics_local"] == 1.0      # noise is in the reads, not in who computes
    assert len(res["cells"]) == 3                          # control + independent + common
