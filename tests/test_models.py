"""Tests for the instance model and the exact formulations."""

from __future__ import annotations

import pytest

from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.metrics import gap_pct, objective_value
from bapbench.models.bap import BAPDisjunctive, BAPTimeIndexed, safe_big_m


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("pytest_toy", 5, 3, seed=20261014)


@pytest.fixture(scope="module")
def hard():
    return make_hard_instance("pytest_hard", 8, 3, seed=20261014, slack=1.8)


def test_generated_instances_are_feasible():
    for inst in (make_toy_instance("t", 5, 3, seed=1),
                 make_toy_instance("t_tight", 6, 3, seed=1, tight=True),
                 make_hard_instance("h", 8, 3, seed=1, slack=1.8)):
        assert inst.validate() == []


def test_tight_berths_still_fit_every_vessel():
    inst = make_toy_instance("t", 6, 3, seed=7, tight=True)
    longest = max(v.length for v in inst.vessels)
    assert all(b.length >= longest for b in inst.berths)
    assert inst.berths[0].length < 2 * longest


def test_slack_controls_the_horizon():
    loose = make_hard_instance("loose", 10, 3, seed=1, slack=2.5)
    tight = make_hard_instance("tight", 10, 3, seed=1, slack=1.2)
    assert loose.horizon > tight.horizon
    assert loose.meta["slack"] == 2.5


def test_tidal_windows_mark_periods_blocked():
    inst = make_hard_instance("h", 10, 3, seed=1, slack=1.8)
    blocked = [(k, t) for k in range(inst.m) for t in range(inst.horizon)
               if inst.blocked(k, t)]
    assert blocked
    assert all(0 <= t < inst.horizon for _k, t in blocked)


def test_crane_capacity_drops_inside_a_failure_window():
    inst = make_hard_instance("h", 10, 3, seed=1, slack=1.8)
    limit = inst.quay_crane_limit
    inst.crane_failures = [(10, 20, 1)]
    assert inst.crane_capacity(5) == limit
    assert inst.crane_capacity(15) == limit - 1
    assert inst.crane_capacity(25) == limit
    assert inst.crane_capacity(5) is not None


def test_both_formulations_agree(toy):
    ti = BAPTimeIndexed(toy).solve()
    dj = BAPDisjunctive(toy).solve()
    assert ti.feasible and dj.feasible
    assert ti.objective == pytest.approx(dj.objective)


@pytest.mark.parametrize("slack", [2.0, 1.8, 1.6])
def test_both_formulations_agree_with_closures_and_cranes(slack):
    """The disjunctive model must honour closures and the crane limit.

    It once encoded only non-overlap and quay capacity, so on a constrained
    instance it returned a schedule that failed the independent check while
    agreeing with the time-indexed model on unrestricted toy instances.
    """
    inst = make_hard_instance(f"both_{slack}", 6, 3, seed=20261014, slack=slack)
    assert inst.tidal_windows and inst.quay_crane_limit is not None

    ti = BAPTimeIndexed(inst).solve()
    dj = BAPDisjunctive(inst).solve()
    assert dj.feasible, dj.violations
    assert ti.feasible and ti.objective == pytest.approx(dj.objective)

    from bapbench.instances import Schedule
    assert Schedule(assign=dict(dj.assign)).check(inst).feasible


def test_disjunctive_rejects_an_instance_with_no_unblocked_window():
    """A vessel with every berth closed for its whole stay cannot be scheduled."""
    inst = make_toy_instance("blocked", 3, 2, seed=5, horizon=12)
    inst.tidal_windows = {str(k): [(0, inst.horizon)] for k in range(inst.m)}
    with pytest.raises(RuntimeError):
        BAPDisjunctive(inst).build()


def test_solution_respects_arrival_times(hard):
    sch = BAPTimeIndexed(hard).solve()
    assert sch.feasible
    for vid, (bid, start) in sch.assign.items():
        assert start >= hard.vessels[vid].arrive
        assert start + hard.vessels[vid].handle <= hard.horizon
        assert not any(hard.blocked(bid, t)
                       for t in range(start, start + hard.vessels[vid].handle))


def test_objective_is_never_below_the_arrival_bound(toy):
    sch = BAPTimeIndexed(toy).solve()
    lower = sum(v.arrive + v.handle for v in toy.vessels)
    assert sch.objective >= lower


def test_safe_big_m_exceeds_any_time_span(toy):
    assert safe_big_m(toy) > toy.horizon


def test_objective_conventions():
    completion = {0: 5, 1: 9, 2: 12}
    assert objective_value(completion, "sum_C") == 26.0
    assert objective_value(completion, "C_max") == 12.0
    with pytest.raises(ValueError):
        objective_value(completion, "nope")


def test_gap_is_undefined_rather_than_infinite():
    assert gap_pct(110.0, 100.0) == 10.0
    assert gap_pct(100.0, 0) is None
    assert gap_pct(None, 100.0) is None


def test_schedule_check_catches_a_bad_start(hard):
    from bapbench.instances import Schedule

    sch = BAPTimeIndexed(hard).solve()
    vid = next(iter(sch.assign))
    bid, _start = sch.assign[vid]
    broken = Schedule(assign={**sch.assign, vid: (bid, 0)})
    broken.check(hard)
    if hard.vessels[vid].arrive > 0:
        assert not broken.feasible
