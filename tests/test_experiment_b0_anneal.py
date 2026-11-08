# tests/test_experiment_b0_anneal.py
from experiment_b0_anneal import run


def test_run_emits_valid_verdicts_and_is_physics_local():
    res = run({"rows": 3, "cols": 3, "n_samples": 3, "epochs": 6, "seeds": [0], "sigma_by_beta": {"0.05": 1e-3}})
    assert res["verdict"]["0.05"]["status"] in {"NOISE-HELPS", "NOISE-NEUTRAL", "NOISE-HURTS"}
    assert res["audit_fraction_physics_local"] == 1.0
    assert [c["arm"] for c in res["cells"]] == ["clean", "constant", "annealed"]
