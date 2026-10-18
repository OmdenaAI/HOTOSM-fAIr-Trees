import numpy as np
import pytest

from fair_trees.evaluate import Counts, bootstrap_ci, paired_bootstrap_diff, prf


def _units(seed=0, k=30):
    rng = np.random.default_rng(seed)
    return [Counts(int(t), int(f), int(n)) for t, f, n in rng.integers(0, 20, (k, 3))]


def test_ci_contains_point_and_is_seeded():
    units = _units()
    lo, hi = bootstrap_ci(units, n=500, seed=0)
    point = prf(sum(units, Counts(0, 0, 0)))[2]
    assert lo <= point <= hi and hi > lo
    assert bootstrap_ci(units, n=500, seed=0) == (lo, hi)


def test_bootstrap_empty():
    assert bootstrap_ci([], n=100) == (0.0, 0.0)
    assert bootstrap_ci([Counts(0, 0, 0)] * 5, n=100) == (0.0, 0.0)


def test_paired_difference():
    a = _units()
    assert paired_bootstrap_diff(a, a, n=300) == (0.0, 0.0, 0.0)
    b = [Counts(c.tp - min(c.tp, 3), c.fp + 3, c.fn + min(c.tp, 3)) for c in a]
    point, lo, hi = paired_bootstrap_diff(a, b, n=300)
    assert point > 0 and lo > 0 and lo <= point <= hi


def test_paired_requires_same_units():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(_units(k=3), _units(k=4))
