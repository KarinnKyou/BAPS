"""Regression tests for the lower-bound references.

A lower bound that exceeds the true optimum turns every gap negative, which is
both wrong and visually obvious; these tests pin the bound below the optimum for
both objectives.
"""

from __future__ import annotations

import pytest

from bapbench.bounds import (
    best_lower_bound,
    lp_lower_bound,
    reference_for,
    trivial_lower_bound,
)
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.models.bap import BAPTimeIndexed


@pytest.mark.parametrize("objective", ["sum_C", "C_max"])
@pytest.mark.parametrize("n,m", [(5, 3), (6, 3), (8, 3)])
def test_lower_bounds_never_exceed_the_optimum(objective, n, m):
    inst = make_toy_instance(f"lb_{n}x{m}", n, m, seed=11)
    exact = BAPTimeIndexed(inst, objective=objective, time_limit=60.0).solve()
    assert exact.extra.get("proved_optimal"), "test needs a proved optimum"

    for bound in (trivial_lower_bound(inst, objective),
                  lp_lower_bound(inst, objective, time_limit=30.0),
                  best_lower_bound(inst, objective, time_limit=30.0)):
        assert bound.value is not None, bound.detail
        assert bound.value <= exact.objective + 1e-6, (
            f"{bound.kind} bound {bound.value} exceeds the {objective} optimum "
            f"{exact.objective}")


def test_trivial_bound_uses_max_for_makespan_not_sum():
    """For C_max the sum of earliest completions is not a bound at all."""
    inst = make_toy_instance("cmax_trivial", 6, 3, seed=1)
    as_sum = trivial_lower_bound(inst, "sum_C").value
    as_max = trivial_lower_bound(inst, "C_max").value
    assert as_max is not None and as_sum is not None
    assert as_max < as_sum
    exact = BAPTimeIndexed(inst, objective="C_max", time_limit=60.0).solve()
    assert as_max <= exact.objective + 1e-6


def test_trivial_bound_rejects_an_unknown_objective():
    inst = make_toy_instance("cmax_bad", 4, 2, seed=1)
    with pytest.raises(ValueError):
        trivial_lower_bound(inst, "not_an_objective")


def test_reference_falls_back_to_a_bound_when_not_proved():
    """An unproved exact value must not be used as the reference."""
    inst = make_toy_instance("ref", 5, 3, seed=3)
    value, kind = reference_for(inst, "sum_C", exact_value=1.0,
                                exact_proved=False, time_limit=30.0)
    assert kind in ("lp_bound", "trivial_bound")
    assert value is not None and value > 1.0

    value, kind = reference_for(inst, "sum_C", exact_value=1.0,
                                exact_proved=True, time_limit=30.0)
    assert kind == "exact_proved" and value == 1.0


def test_lp_bound_is_valid_on_a_constrained_instance():
    inst = make_hard_instance("lb_hard", 8, 3, seed=20261014, slack=1.8)
    exact = BAPTimeIndexed(inst, objective="sum_C", time_limit=60.0).solve()
    assert exact.extra.get("proved_optimal")
    bound = lp_lower_bound(inst, "sum_C", time_limit=30.0)
    assert bound.value is not None
    assert bound.value <= exact.objective + 1e-6
