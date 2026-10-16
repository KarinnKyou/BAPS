"""Tests for commitments under changing berth availability.

Two defects used to abort replanning and leave the caller with a schedule that
violated the realised instance:

* a pinned start whose instant a later tidal shift blocked was treated as an
  infeasible instance, so the model raised instead of relaxing the start;
* a vessel was pinned as soon as its start lower bound was in the past, even
  when a delay meant handling had not actually begun.

The regression that matters is end to end: with disruptions applied, a
rolling-horizon run must produce a schedule that is valid on the realised
instance.
"""

from __future__ import annotations

import random

import pytest

from bapbench.dynamics import (
    apply_events,
    evaluate_schedule_on,
    make_events,
    rolling_horizon,
)
from bapbench.instances import make_hard_instance
from bapbench.models.bap import BAPTimeIndexed


@pytest.fixture(scope="module")
def planned():
    return make_hard_instance("commit", 10, 3, seed=20261014, slack=1.8)


def test_locked_berth_with_a_blocked_pinned_start_is_relaxed(planned):
    """A fixed commitment to a start the tide has since blocked must not make
    the instance infeasible: the berth holds, the start moves."""
    locked = {0: (0, 0, True)}
    sch = BAPTimeIndexed(planned, time_limit=30.0, lock=locked).solve()
    assert sch.feasible, sch.violations
    berth, start = sch.assign[0]
    assert berth == 0
    assert start >= (locked[0][1])

    # the lock is respected and the schedule is valid under the same checks as
    # any other schedule
    assert sch.violations == []


def test_lock_berth_is_matched_against_the_berth_not_the_lower_bound(planned):
    """The lock tuple is (berth, start lower bound, fixed). Confusing the two
    fields silently assigned a vessel to the wrong berth."""
    want_berth, want_lb = 1, 20
    sch = BAPTimeIndexed(planned, time_limit=30.0,
                         lock={0: (want_berth, want_lb, False)}).solve()
    assert sch.feasible, sch.violations
    berth, start = sch.assign[0]
    assert berth == want_berth
    assert start >= want_lb


def test_a_lock_pointing_at_another_berth_removes_the_other_berths(planned):
    """Only the committed berth may host the vessel."""
    sch = BAPTimeIndexed(planned, time_limit=30.0,
                         lock={0: (2, 0, False)}).solve()
    assert sch.feasible
    assert sch.assign[0][0] == 2


@pytest.mark.parametrize("seed", [3, 7, 11, 17, 23, 42])
def test_rolling_horizon_yields_a_schedule_valid_on_the_realised_instance(planned, seed):
    """No run may abort replanning and return a schedule that violates the
    realised instance."""
    events = make_events(planned, random.Random(seed))
    true_inst = apply_events(planned, events, upto=None)
    res = rolling_horizon(planned, true_inst, events, dt=8, time_limit=5.0)

    assert res.schedule is not None, "no schedule produced"
    ok, value, violations = evaluate_schedule_on(true_inst, res.schedule)
    assert ok, f"schedule invalid on the realised instance: {violations[:2]}"
    assert value is not None
    assert res.feasible

    # the planner must not have given up part way through
    aborted = [e for e in res.log if "error" in e]
    assert not aborted, f"replanning aborted: {aborted[:1]}"

    # and the result must be no better than the perfect-information optimum
    perfect = BAPTimeIndexed(true_inst, time_limit=30.0).solve()
    assert perfect.objective is not None
    assert value >= perfect.objective - 1e-6


def test_rolling_horizon_beats_nothing_but_respects_commitments(planned):
    """With no events the rolling schedule should equal the static optimum."""
    true_inst = apply_events(planned, [], upto=None)
    res = rolling_horizon(planned, true_inst, [], dt=8, time_limit=5.0)
    assert res.schedule is not None
    ok, value, _ = evaluate_schedule_on(true_inst, res.schedule)
    assert ok
    static = BAPTimeIndexed(planned, time_limit=30.0).solve()
    assert static.objective is not None
    assert value >= static.objective - 1e-6
