# tests/test_experiment_b0.py
from experiment_b0 import run, _task_verdict, _converged


def test_task_verdict_is_pass_only_if_every_seed_passes_else_most_severe():
    mk = lambda s: {"verdict": {"status": s}}
    assert _task_verdict([mk("PASS"), mk("PASS")]) == "PASS"
    assert _task_verdict([mk("PASS"), mk("AMBIGUOUS")]) == "AMBIGUOUS"
    assert _task_verdict([mk("AMBIGUOUS"), mk("MULTISTABILITY-FORECLOSE"), mk("PASS")]) == "MULTISTABILITY-FORECLOSE"
    assert _converged([1.0] * 50) is True and _converged(list(range(50, 0, -1))) is False


def test_run_emits_valid_verdicts_and_headline_is_physics_local():
    res = run({"rows": 3, "cols": 3, "n_samples": 3, "factors": {"amplify": 2.0}, "seeds": [0],
               "epochs": 30, "ceiling_maxiter": 30, "d_map": [0.1], "d_map_epochs": 10})
    t = res["tasks"]["amplify"]
    assert t["verdict"] in {"PASS", "FORECLOSE-REPRESENTABILITY", "FORECLOSE-LEARNABILITY",
                            "MULTISTABILITY-FORECLOSE", "AMBIGUOUS"}   # a VALID verdict, never which
    assert res["audit_fraction_physics_local"] == 1.0                 # ceiling/baseline logs are separate
    assert len(res["d_map"]) == 1 and "untrained_ratio" in res
