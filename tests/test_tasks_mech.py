import numpy as np
import pytest
from tasks import parity3_mech_task, two_sided_separation, verdict_b


def test_parity3_task_structure():
    task = parity3_mech_task()
    assert task["patterns"].shape == (8, 3)
    assert set(np.unique(task["patterns"])) <= {-1.0, 1.0}
    assert task["labels"].shape == (8,)
    # parity label = product of bits
    assert task["labels"] == pytest.approx(np.prod(task["patterns"], axis=1))
    # input/output DOFs are distinct and exist
    net = task["net"]
    assert len(net.input_dofs) == 3
    assert net.output_dof not in net.input_dofs
    assert net.output_dof not in net.pinned_dofs


def test_two_sided_separation():
    y = np.array([1, 1, -1, -1])
    assert two_sided_separation(np.array([2.0, 3.0, 0.0, 1.0]), y) == pytest.approx(1.0)   # +class above
    assert two_sided_separation(np.array([0.0, 1.0, 2.0, 3.0]), y) == pytest.approx(1.0)   # either polarity
    assert two_sided_separation(np.array([0.0, 2.0, 1.0, 3.0]), y) < 0                     # overlap


def test_two_sided_separation_single_class():
    import numpy as np
    from tasks import two_sided_separation
    assert two_sided_separation(np.array([0.3, 0.4]), np.array([1, 1])) == float("-inf")
    assert two_sided_separation(np.array([0.3, 0.4]), np.array([-1, -1])) == float("-inf")


def test_verdict_b_branches():
    # multistability gate fires first regardless of separation
    assert verdict_b(1.0, 1.0, 0.1, jump_rate=0.5, sign_match=1.0, rho=1.0)["status"] == "MULTISTABILITY-FORECLOSE"
    # baseline can't separate -> representability foreclosure
    assert verdict_b(-0.1, -0.1, 0.1, 0.0, 1.0, 1.0)["status"] == "FORECLOSE-REPRESENTABILITY"
    # baseline separates, local rule can't -> learnability foreclosure
    assert verdict_b(-0.1, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "FORECLOSE-LEARNABILITY"
    # strong separation, aligned, no jumps -> PASS
    assert verdict_b(1.0, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "PASS"
    # positive but sub-SNR -> AMBIGUOUS
    assert verdict_b(0.15, 1.0, 0.1, 0.0, 1.0, 1.0)["status"] == "AMBIGUOUS"


# --- verdict_b2 branches -----------------------------------------------------------------------
from tasks import verdict_b2


def _row(d, sep_min, sep_max, unstable=0.0, sat=0.0, floor=1e-3):
    return {"d": d, "sep_min": sep_min, "sep_max": sep_max, "snr_min": sep_min / floor,
            "unstable_frac": unstable, "saturation_gap": sat}


def test_verdict_b2_representable_needs_a_robust_stable_positive():
    assert verdict_b2([_row(0.1, 0.01, 0.02)])["status"] == "REPRESENTABLE"
    # positive only on the best seed -> not robust
    assert verdict_b2([_row(0.1, -0.01, 0.02)])["status"] == "AMBIGUOUS"
    # robust positive but only in the unstable regime
    assert verdict_b2([_row(0.3, 0.01, 0.02, unstable=0.5)])["status"] == "AMBIGUOUS"


def test_verdict_b2_confirmed_foreclose_requires_saturation_and_stability():
    rows = [_row(0.05, -0.1, -0.05), _row(0.1, -0.2, -0.1)]
    assert verdict_b2(rows)["status"] == "FORECLOSE-REPRESENTABILITY-CONFIRMED"
    rows_unsat = [_row(0.05, -0.1, -0.05, sat=0.1)]
    assert verdict_b2(rows_unsat)["status"] == "AMBIGUOUS"          # budget not saturated
    assert verdict_b2([_row(0.3, -0.1, -0.05, unstable=0.9)])["status"] == "AMBIGUOUS"  # nothing stable


# --- Experiment B0: mechanical allostery task + verdict_b0 --------------------------------------
from tasks import allostery_mech_task, verdict_b0


def test_allostery_mech_task_structure():
    t = allostery_mech_task(4, 4, d_max=0.1, ratio=0.5, n_samples=6)
    assert t["patterns"].shape == (6, 1) and t["targets"].shape == (6,)
    assert np.all(t["patterns"] > 0) and np.all(t["patterns"] <= 0.1)
    assert t["targets"] == pytest.approx(0.5 * t["patterns"][:, 0])
    assert t["labels"] is None and t["d"] == 1.0
    assert len(t["net"].input_dofs) == 1 and len(t["net"].pinned_dofs) == 3


def test_verdict_b0_branches_in_order():
    ok = dict(sign_match=1.0, rho=1.0)
    assert verdict_b0(1e-4, 1e-4, 1.0, 0.0, **ok)["status"] == "PASS"
    assert verdict_b0(5e-4, 1e-9, 1.0, 0.0, **ok)["status"] == "PASS"          # abs_floor saves a tiny baseline
    assert verdict_b0(1e-4, 1e-4, 1.0, 0.5, **ok)["status"] == "MULTISTABILITY-FORECLOSE"
    assert verdict_b0(1e-4, 0.5, 1.0, 0.0, **ok)["status"] == "FORECLOSE-REPRESENTABILITY"
    # the representability gate is the converged ceiling, not the matched-budget baseline
    assert verdict_b0(0.3, 0.1, 1.0, 0.0, ceiling_error=1e-3, **ok)["status"] == "AMBIGUOUS"
    assert verdict_b0(1e-4, 1e-4, 1.0, 0.0, ceiling_error=0.5, **ok)["status"] == "FORECLOSE-REPRESENTABILITY"
    assert verdict_b0(0.8, 1e-4, 1.0, 0.0, **ok)["status"] == "FORECLOSE-LEARNABILITY"
    assert verdict_b0(0.1, 1e-4, 1.0, 0.0, **ok)["status"] == "AMBIGUOUS"       # learns, misses baseline
    assert verdict_b0(1e-4, 1e-4, 1.0, 0.0, sign_match=0.5, rho=0.2)["status"] == "AMBIGUOUS"
