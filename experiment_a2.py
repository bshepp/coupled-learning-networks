# experiment_a2.py
"""Experiment A2: two completeness items left open by Experiment A's findings.

(1) The beta-tradeoff, made visible. A's 3x3 uniform-init net showed rho already 0.9996 at beta = 0.2, so
    the equilibrium-propagation beta -> 0 limit was asserted but barely exhibited. Here a HETEROGENEOUS
    5x5 resistor net (random conductances) is swept over beta and 1 - rho is fitted as a power law; for
    a linear network the update's deviation from the gradient is first order in beta, so 1 - rho (an
    angle squared) should scale as beta^2, with no numerical floor until beta ~ machine precision.
(2) The Hagen-Poiseuille law (g = C r^4, trainable radius r), implemented since A but never run in a
    headline: A's two tasks, A's verdict() thresholds unchanged, radius clipped to [0.3, 2.0]. This is
    the flow-network form that 07 (the hardware-transfer arm) builds on.
Audit: headline runs PHYSICS_LOCAL; baselines OFFLINE_GRADIENT (separate logs), exactly as in A."""
import json
import time
import numpy as np
from network import RESISTOR, poiseuille_law, grid_network
from tasks import regression_task, allostery_task, Task, verdict
from coupled_learning import train, free_phase, nudged_phase, local_update
from baselines import gd_baseline, adjoint_gradient, gradient_alignment
from audit import AuditLog

DEFAULT_CONFIG = {"sweep_rows": 5, "sweep_cols": 5, "sweep_seed": 3, "g_range": (0.2, 5.0),
                  "betas": [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 1e-4, 1e-5, 1e-6],
                  "epochs": 400, "baseline_epochs": 400, "beta": 0.01, "eta": 0.5, "lr": 0.5,
                  "r_min": 0.3, "r_max": 2.0, "C": 1.0,
                  # A2-specific verdict settings (A's own thresholds are unchanged): a normalized MSE
                  # <= abs_floor counts as reached (both runs sit at machine zero otherwise), and the
                  # sign-match excludes exactly-zero-gradient edges of the symmetric lattice
                  "abs_floor": 1e-6}


def beta_sweep(cfg):
    net = grid_network(cfg["sweep_rows"], cfg["sweep_cols"], n_in=2, n_out=1)
    rng = np.random.default_rng(cfg["sweep_seed"])
    X = rng.uniform(0.2, 1.0, size=(8, 2)); Y = (X @ np.array([0.4, 0.3])).reshape(-1, 1)
    task = Task("regression-heterogeneous", net, X, Y)
    g = rng.uniform(*cfg["g_range"], len(net.edges))
    neg_grad = -adjoint_gradient(net, g, task)
    rows = []
    for beta in cfg["betas"]:
        Vf = free_phase(net, g, X)
        Vn = nudged_phase(net, g, X, Y, beta, free_out=Vf[:, net.outputs])
        a = gradient_alignment(local_update(net, g, RESISTOR, Vf, Vn, beta, eta=1.0), neg_grad)
        rows.append({"beta": beta, "sign_match": a["sign_match"], "rho": a["rho"], "one_minus_rho": 1.0 - a["rho"]})
    pts = [(r["beta"], r["one_minus_rho"]) for r in rows if r["one_minus_rho"] > 1e-12]
    slope = float(np.polyfit(np.log([b for b, _ in pts]), np.log([v for _, v in pts]), 1)[0]) if len(pts) >= 3 else float("nan")
    return {"rows": rows, "power_law_exponent": slope, "n_fit_points": len(pts)}


def _one_task(task, law, cfg):
    net = task.net
    p0 = np.full(len(net.edges), 1.0)
    audit = AuditLog()
    local = train(net, p0, law, task, cfg["beta"], cfg["eta"], cfg["epochs"],
                  p_min=cfg["r_min"], p_max=cfg["r_max"], audit=audit)
    base = gd_baseline(net, p0, law, task, cfg["baseline_epochs"], cfg["lr"], p_min=cfg["r_min"], p_max=cfg["r_max"])
    # alignment at INIT (commons alignment-certification), chain rule through dg/dr
    g0 = law.g_of_p(p0)
    Vf = free_phase(net, g0, task.inputs)
    Vn = nudged_phase(net, g0, task.inputs, task.targets, cfg["beta"], free_out=Vf[:, net.outputs])
    dp = local_update(net, p0, law, Vf, Vn, cfg["beta"], eta=1.0)
    align = gradient_alignment(dp, -(adjoint_gradient(net, g0, task) * law.dg_dp(p0)))
    v = verdict(local["final_error"], base["final_error"], align["sign_match_significant"], align["rho"],
                abs_floor=cfg["abs_floor"])
    return {"local_error": local["final_error"], "baseline_error": base["final_error"],
            "sign_match": align["sign_match"], "sign_match_significant": align["sign_match_significant"],
            "n_significant": align["n_significant"], "rho": align["rho"], "verdict": v,
            "radius_range": [float(local["params"].min()), float(local["params"].max())],
            "audit_fraction_physics_local": audit.summary()["fraction_physics_local"]}


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    t0 = time.time()
    law = poiseuille_law(cfg["C"])
    out = {"config": cfg, "beta_sweep": beta_sweep(cfg), "poiseuille": {}}
    out["poiseuille"]["allostery"] = _one_task(allostery_task(), law, cfg)
    out["poiseuille"]["regression"] = _one_task(regression_task(), law, cfg)
    out["audit"] = {"fraction_physics_local": min(v["audit_fraction_physics_local"] for v in out["poiseuille"].values())}
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    res = run()
    bs = res["beta_sweep"]
    print("beta-sweep, heterogeneous 5x5 resistor net (alignment of the local update with -grad L):")
    print(f"{'beta':>9}{'sign':>7}{'rho':>12}{'1-rho':>11}")
    for r in bs["rows"]:
        print(f"{r['beta']:>9.0e}{r['sign_match']:>7.3f}{r['rho']:>12.8f}{r['one_minus_rho']:>11.2e}")
    print(f"power-law fit: 1 - rho ~ beta^{bs['power_law_exponent']:.2f}  ({bs['n_fit_points']} points)")
    print(f"\nPoiseuille law (g = r^4), A's tasks and verdict:")
    print(f"{'task':<12}{'local_err':>12}{'base_err':>12}{'sign':>7}{'sign*':>7}{'rho':>8}{'r range':>16}{'verdict':>12}")
    for name, r in res["poiseuille"].items():
        print(f"{name:<12}{r['local_error']:>12.3e}{r['baseline_error']:>12.3e}{r['sign_match']:>7.2f}"
              f"{r['sign_match_significant']:>7.2f}{r['rho']:>8.3f}   [{r['radius_range'][0]:.2f}, {r['radius_range'][1]:.2f}]"
              f"{r['verdict']['status']:>12}")
    print("  (sign* = sign-match over edges with a non-zero gradient; the symmetric lattice has exactly-zero ones)")
    print(f"\nphysics-local = {res['audit']['fraction_physics_local']:.2f}  ({res['elapsed_s']:.0f}s)")
    with open("experiment_a2_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote experiment_a2_results.json")


if __name__ == "__main__":
    main()
