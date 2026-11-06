# tests/test_experiment_a2.py
from experiment_a2 import run


def test_run_smoke_valid_verdicts_physics_local_and_quadratic_beta_law():
    res = run({"epochs": 100, "baseline_epochs": 100, "betas": [0.2, 0.05, 0.01, 0.002]})
    for name in ("allostery", "regression"):
        r = res["poiseuille"][name]
        assert r["verdict"]["status"] in ("PASS", "AMBIGUOUS", "FORECLOSE")   # valid, never which
        assert r["audit_fraction_physics_local"] == 1.0
    # linear net: the update's deviation from the gradient is O(beta) -> 1 - rho ~ beta^2
    assert abs(res["beta_sweep"]["power_law_exponent"] - 2.0) < 0.2
    rhos = [r["rho"] for r in res["beta_sweep"]["rows"]]
    assert rhos == sorted(rhos)                                               # monotone in shrinking beta
