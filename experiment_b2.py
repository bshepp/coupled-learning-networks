# experiment_b2.py
"""Experiment B2: is Experiment B's representability FORECLOSE a budget / objective artifact?

B declared FORECLOSE-REPRESENTABILITY from an offline baseline that (i) minimized MSE to FIXED-scale
targets (+/-0.5 d) with (ii) plain gradient descent for (iii) a fixed 400-epoch budget. The commons lesson
`foreclosure-can-be-a-budget-artifact` says such a foreclosure is only valid if the ceiling is
budget-saturated and the objective does not presuppose a scale. B2 re-probes representability with:
  * a SCALE-FREE objective -- the Walsh parity purity a_123^2 / sum_j a_j^2 (1 iff the readout IS parity
    up to offset and scale), differentiated through the implicit adjoint;
  * a better-converged optimizer (L-BFGS-B on the adjoint) with a budget-saturation record, plus the
    original MSE objective under L-BFGS-B as a control, plus a gradient-free differential-evolution
    probe (optimizer class);
  * the Walsh spectrum of the UNTRAINED lattice vs d (how the parity channel scales with deformation).
Every offline update is tagged OFFLINE_GRADIENT -- this is a representability probe, not a learning run.
The outcome (REPRESENTABLE / FORECLOSE-REPRESENTABILITY-CONFIRMED / AMBIGUOUS) is computed, not assumed."""
import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from spring_network import default_rest_lengths
from tasks import parity3_mech_task, two_sided_separation, verdict_b2
from mechanical_learning import free_phase_mech, readout
from baselines import mech_readout_adjoint
from parity_analysis import parity_purity, parity_purity_grad, spectrum_report
from audit import AuditLog, OFFLINE_GRADIENT

DEFAULT_CONFIG = {"rows": 4, "cols": 4, "ds": [0.05, 0.10, 0.15, 0.20], "seeds": [0, 1, 2],
                  "budgets": [50, 200, 800], "floor": 1e-3,
                  "k_bounds": (1e-2, 1e2), "l_bounds": (0.1, 5.0),
                  "de": {"d": 0.15, "seed": 0, "maxiter": 30, "popsize": 6,
                         "k_bounds": (0.1, 10.0), "l_rel_bounds": (0.6, 1.4)}}
FAIL_LOSS = 2.0     # worse than any purity loss (<= 1) or reachable MSE; returned when a solve fails


class Probe:
    """Offline representability probe on one task. Counts solve failures (instability events)."""

    def __init__(self, task, audit=None):
        self.task = task; self.net = task["net"]
        self.S = task["patterns"]; self.y = task["labels"]; self.d = task["d"]
        self.targets = np.where(self.y > 0, task["target_pos"], task["target_neg"])
        self.E = len(self.net.edges)
        self.audit = audit
        self.n_eval = 0; self.n_fail = 0

    def split(self, theta):
        return np.asarray(theta[:self.E], float), np.asarray(theta[self.E:], float)

    def readouts(self, k, l):
        xs = [free_phase_mech(self.net, k, l, self.S[p], self.d)[0] for p in range(len(self.S))]
        return xs, np.array([readout(self.net, x) for x in xs])

    def _loss_grad(self, theta, dloss_dr, loss_of_r):
        k, l = self.split(theta)
        self.n_eval += 1
        try:
            xs, r = self.readouts(k, l)
            gk, gl = mech_readout_adjoint(self.net, k, l, self.S, self.d, dloss_dr(r), xs=xs)
        except RuntimeError:
            self.n_fail += 1
            return FAIL_LOSS, np.zeros_like(theta)
        if self.audit is not None:
            self.audit.record(OFFLINE_GRADIENT, 2 * self.E)
        return float(loss_of_r(r)), np.concatenate([gk, gl])

    def purity_loss(self, theta):
        return self._loss_grad(theta, lambda r: -parity_purity_grad(r, self.S),
                               lambda r: 1.0 - parity_purity(r, self.S))

    def mse_loss(self, theta):
        P = len(self.S)
        return self._loss_grad(theta, lambda r: (r - self.targets) / P,
                               lambda r: 0.5 * np.mean((r - self.targets) ** 2))

    def purity_only(self, theta):
        k, l = self.split(theta)
        self.n_eval += 1
        try:
            _, r = self.readouts(k, l)
        except RuntimeError:
            self.n_fail += 1
            return FAIL_LOSS
        return 1.0 - parity_purity(r, self.S)

    def score(self, theta):
        """Verdict quantities at a configuration (None if its equilibria cannot be solved)."""
        k, l = self.split(theta)
        try:
            _, r = self.readouts(k, l)
        except RuntimeError:
            return None
        rep = spectrum_report(r, self.S)
        rep["sep"] = two_sided_separation(r, self.y)
        rep["readouts"] = r.tolist()
        return rep


def _init(task, seed):
    net = task["net"]; rng = np.random.default_rng(seed)
    k0 = rng.uniform(0.5, 2.0, len(net.edges))
    l0 = default_rest_lengths(net) * rng.uniform(0.9, 1.1, len(net.edges))
    return np.concatenate([k0, l0])


def lbfgs_probe(probe, theta0, loss_fn, budgets, k_bounds, l_bounds):
    """L-BFGS-B from theta0 for budgets[-1] iterations, recording the purity at each intermediate
    budget (one deterministic run; its trajectory at iteration b equals an independent run of budget b)."""
    bounds = [k_bounds] * probe.E + [l_bounds] * probe.E
    trace = {}
    state = {"it": 0}

    def cb(xk):
        state["it"] += 1
        if state["it"] in budgets:
            sc = probe.score(xk)
            trace[state["it"]] = (sc["purity"] if sc else None)

    res = minimize(loss_fn, theta0, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                   options={"maxiter": budgets[-1], "maxfun": 4 * budgets[-1]})
    sc = probe.score(res.x)
    trace[budgets[-1]] = (sc["purity"] if sc else None)
    return {"theta": res.x, "nit": int(res.nit), "message": str(res.message), "trace": trace,
            "score": sc}


def de_probe(probe, cfg_de, seed):
    """Gradient-free differential evolution on the purity objective (optimizer-class check)."""
    l_def = default_rest_lengths(probe.net)
    bounds = [cfg_de["k_bounds"]] * probe.E + [(cfg_de["l_rel_bounds"][0] * ld, cfg_de["l_rel_bounds"][1] * ld)
                                               for ld in l_def]
    res = differential_evolution(probe.purity_only, bounds, seed=seed, maxiter=cfg_de["maxiter"],
                                 popsize=cfg_de["popsize"], polish=False, tol=0.0)
    if probe.audit is not None:
        probe.audit.record(OFFLINE_GRADIENT, 2 * probe.E * int(res.nfev))
    return {"theta": res.x, "nfev": int(res.nfev), "score": probe.score(res.x)}


def untrained_spectrum(rows, cols, ds):
    """Walsh spectrum of the uniform, unstressed lattice vs d (the substrate's native parity channel)."""
    out = []
    for d in ds:
        task = parity3_mech_task(rows, cols, d=d); net = task["net"]
        k = np.ones(len(net.edges)); l = default_rest_lengths(net)
        try:
            r = np.array([readout(net, free_phase_mech(net, k, l, task["patterns"][p], d)[0])
                          for p in range(8)])
            rep = spectrum_report(r, task["patterns"]); rep["d"] = d; rep["stable"] = True
        except RuntimeError as e:
            rep = {"d": d, "stable": False, "error": str(e)}
        out.append(rep)
    return out


def _aggregate(d, cells, floor):
    seps = [c["score"]["sep"] for c in cells if c["score"] is not None]
    sep_min = min(seps) if seps else float("-inf")
    sep_max = max(seps) if seps else float("-inf")
    n_eval = sum(c["n_eval"] for c in cells); n_fail = sum(c["n_fail"] for c in cells)
    gaps = [c["saturation_gap"] for c in cells if c.get("saturation_gap") is not None]
    return {"d": d, "sep_min": sep_min, "sep_max": sep_max, "snr_min": sep_min / floor,
            "purity_max": max([c["score"]["purity"] for c in cells if c["score"]] or [0.0]),
            "fail_rate": n_fail / max(n_eval, 1),
            "saturation_gap": max(gaps) if gaps else float("inf"), "n_cells": len(cells)}


def run(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    audit = AuditLog()
    budgets = list(cfg["budgets"])
    t0 = time.time()
    out = {"config": {k: v for k, v in cfg.items()}, "untrained_spectrum":
           untrained_spectrum(cfg["rows"], cfg["cols"], sorted(set(cfg["ds"]) | {0.02, 0.3}))}
    cells = []
    for d in cfg["ds"]:
        for seed in cfg["seeds"]:
            task = parity3_mech_task(cfg["rows"], cfg["cols"], d=d, seed=seed)
            theta0 = _init(task, seed)
            for objective in ("purity", "mse"):
                probe = Probe(task, audit=audit)
                loss = probe.purity_loss if objective == "purity" else probe.mse_loss
                res = lbfgs_probe(probe, theta0, loss, budgets, tuple(cfg["k_bounds"]), tuple(cfg["l_bounds"]))
                tr = res["trace"]
                gap = None
                if len(budgets) >= 2 and tr.get(budgets[-1]) is not None and tr.get(budgets[-2]) is not None:
                    gap = tr[budgets[-1]] - tr[budgets[-2]]
                cells.append({"d": d, "seed": seed, "optimizer": "lbfgs", "objective": objective,
                              "nit": res["nit"], "message": res["message"],
                              "purity_trace": {str(b): v for b, v in tr.items()},
                              "saturation_gap": gap, "n_eval": probe.n_eval, "n_fail": probe.n_fail,
                              "score": res["score"]})
    if cfg.get("de"):
        de = cfg["de"]
        task = parity3_mech_task(cfg["rows"], cfg["cols"], d=de["d"], seed=de["seed"])
        probe = Probe(task, audit=audit)
        res = de_probe(probe, de, de["seed"])
        cells.append({"d": de["d"], "seed": de["seed"], "optimizer": "de", "objective": "purity",
                      "nfev": res["nfev"], "saturation_gap": None,
                      "n_eval": probe.n_eval, "n_fail": probe.n_fail, "score": res["score"]})
    rows = [_aggregate(d, [c for c in cells if c["d"] == d], cfg["floor"]) for d in sorted({c["d"] for c in cells})]
    out["cells"] = cells
    out["rows"] = rows
    out["verdict"] = verdict_b2(rows)
    out["audit"] = audit.summary()
    out["audit"]["all_offline"] = audit.summary()["counts"].get(OFFLINE_GRADIENT, 0) == audit.summary()["total"]
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    res = run()
    print("untrained lattice Walsh spectrum vs d:")
    print(f"{'d':>6}{'linear':>12}{'quad':>12}{'parity':>12}{'parity/lin':>12}{'purity':>10}")
    for s in res["untrained_spectrum"]:
        if s["stable"]:
            print(f"{s['d']:>6.2f}{s['linear']:>12.3e}{s['quadratic']:>12.3e}{s['parity']:>12.3e}"
                  f"{s['parity_to_linear']:>12.3e}{s['purity']:>10.4f}")
        else:
            print(f"{s['d']:>6.2f}  UNSTABLE (buckled)")
    print("\noffline probes (per cell):")
    print(f"{'d':>6}{'seed':>5}{'opt':>7}{'obj':>8}{'purity':>9}{'sep':>12}{'fail':>6}{'trace':>30}")
    for c in res["cells"]:
        sc = c["score"]
        pur = f"{sc['purity']:.4f}" if sc else "  n/a"
        sep = f"{sc['sep']:.4e}" if sc else "    n/a"
        tr = " ".join(f"{k}:{v:.3f}" if v is not None else f"{k}:n/a" for k, v in c.get("purity_trace", {}).items())
        print(f"{c['d']:>6.2f}{c['seed']:>5}{c['optimizer']:>7}{c['objective']:>8}{pur:>9}{sep:>12}"
              f"{c['n_fail']/max(c['n_eval'],1):>6.2f}  {tr}")
    print("\nper-d aggregate:")
    print(f"{'d':>6}{'sep_min':>12}{'sep_max':>12}{'purity_max':>11}{'fail_rate':>10}{'sat_gap':>9}")
    for r in res["rows"]:
        print(f"{r['d']:>6.2f}{r['sep_min']:>12.4e}{r['sep_max']:>12.4e}{r['purity_max']:>11.4f}"
              f"{r['fail_rate']:>10.2f}{r['saturation_gap']:>9.3f}")
    print(f"\nVERDICT: {res['verdict']['status']}  (offline-only audit: {res['audit']['all_offline']}, "
          f"{res['elapsed_s']:.0f}s)")
    with open("experiment_b2_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote experiment_b2_results.json")


if __name__ == "__main__":
    main()
