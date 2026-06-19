# tests/test_experiment_b.py
from experiment_b import run


def test_run_emits_a_valid_verdict_and_is_physics_local():
    res = run({"rows": 3, "cols": 3, "epochs": 60, "baseline_epochs": 60,
               "beta": 0.02, "eta_k": 1.0, "eta_l": 1.0, "lr": 1.0, "seeds": [0, 1], "d": 0.1})
    assert res["verdict"]["status"] in {
        "MULTISTABILITY-FORECLOSE", "FORECLOSE-REPRESENTABILITY",
        "FORECLOSE-LEARNABILITY", "PASS", "AMBIGUOUS"}     # a VALID verdict, never asserting which
    assert res["audit_fraction_physics_local"] == 1.0     # headline training is honest
