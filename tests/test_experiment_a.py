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
    assert res["audit"]["fraction_physics_local"] == 1.0
