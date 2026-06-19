# experiment_a.py
"""Driver for Experiment A: train the resistor network by the local coupled-learning rule
on allostery + regression, compare to the offline adjoint-gradient baseline, measure
gradient alignment, and emit the pre-registered verdict. Answers 'who computes the update?'
with a logged, asserted audit trail."""
import json
import numpy as np
from network import RESISTOR
from tasks import regression_task, allostery_task, verdict
from coupled_learning import train, free_phase, nudged_phase, local_update
from baselines import gd_baseline, adjoint_gradient, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"epochs": 400, "baseline_epochs": 400, "beta": 0.01, "eta": 0.5, "lr": 0.5}


def _one_task(task, cfg):
    net = task.net
    params0 = np.full(len(net.edges), 1.0)

    audit = AuditLog()
    local = train(net, params0, RESISTOR, task, cfg["beta"], cfg["eta"], cfg["epochs"], audit=audit)

    base = gd_baseline(net, params0, RESISTOR, task, cfg["baseline_epochs"], cfg["lr"])

    g = RESISTOR.g_of_p(local["params"])
    Vf = free_phase(net, g, task.inputs)
    Vn = nudged_phase(net, g, task.inputs, task.targets, cfg["beta"], free_out=Vf[:, net.outputs])
    dp = local_update(net, g, RESISTOR, Vf, Vn, cfg["beta"], eta=1.0)
    align = gradient_alignment(dp, -adjoint_gradient(net, g, task))

    v = verdict(local["final_error"], base["final_error"], align["sign_match"], align["rho"])
    return {"local_error": local["final_error"], "baseline_error": base["final_error"],
            "sign_match": align["sign_match"], "rho": align["rho"], "verdict": v,
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"]}


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    out = {}
    out["allostery"] = _one_task(allostery_task(), cfg)
    out["regression"] = _one_task(regression_task(), cfg)
    return out


def main():
    res = run()
    print(f"{'task':<12}{'local_err':>12}{'base_err':>12}{'sign':>8}{'rho':>8}{'verdict':>12}")
    for name, r in res.items():
        print(f"{name:<12}{r['local_error']:>12.4e}{r['baseline_error']:>12.4e}"
              f"{r['sign_match']:>8.2f}{r['rho']:>8.3f}{r['verdict']['status']:>12}")
    with open("experiment_a_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote experiment_a_results.json")


if __name__ == "__main__":
    main()
