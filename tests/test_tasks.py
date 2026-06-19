import numpy as np
import pytest
from tasks import task_error, regression_task, allostery_task, verdict


def test_task_error_zero_and_positive():
    t = np.array([[0.4], [0.6]])
    assert task_error(t, t) == pytest.approx(0.0)
    assert task_error(t + 0.1, t) > 0.0


def test_regression_targets_representable():
    # passive network output obeys the maximum principle: target must lie within input range
    task = regression_task()
    assert task.targets.min() >= 0.0
    assert task.targets.max() <= task.inputs.max() + 1e-9


def test_allostery_targets_are_ratio_of_source():
    task = allostery_task(ratio=0.5)
    assert task.targets == pytest.approx(0.5 * task.inputs)


def test_verdict_branches():
    assert verdict(0.01, 0.01, 1.0, 1.0)["status"] == "PASS"
    assert verdict(0.5, 0.01, 1.0, 1.0)["status"] == "FORECLOSE"
    assert verdict(0.01, 0.01, 0.5, 0.5)["status"] == "AMBIGUOUS"
