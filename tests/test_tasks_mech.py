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
