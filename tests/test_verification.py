"""Tests for the verifiers and the disruption model."""

from __future__ import annotations

import random

import pytest

from bapbench.dynamics import (
    Event,
    apply_events,
    evaluate_schedule_on,
    make_events,
    observed_arrivals,
)
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.models.bap import BAPTimeIndexed
from bapbench.verification.benchmarks import (
    case_flow_bottleneck,
    case_parallel_balanced,
    case_single_machine,
    case_two_implementations_agree,
)
from bapbench.verification.exhaustive import exhaustive_optimum, verify_instance


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("pytest_exh", 5, 3, seed=20261014)


@pytest.mark.parametrize("case", [case_single_machine, case_parallel_balanced,
                                  case_flow_bottleneck,
                                  case_two_implementations_agree])
def test_known_optimum_families(case):
    result = case()
    assert result["match_expected"]


def test_enumeration_matches_the_exact_model(toy):
    sch = BAPTimeIndexed(toy).solve()
    agrees, result = verify_instance(toy, sch.objective, max_cases=2_000_000,
                                     time_budget=120.0)
    assert result.objective is not None
    assert agrees


def test_enumeration_refuses_instances_that_are_too_large():
    big = make_toy_instance("big", 12, 4, seed=1)
    result = exhaustive_optimum(big, max_cases=1_000)
    assert result.objective is None
    assert "too large" in result.note


def test_apply_events_does_not_mutate_its_input():
    inst = make_hard_instance("h", 8, 3, seed=1, slack=1.8)
    before = inst.dumps()
    events = [Event("arrival_delay", 0, 0, 4, "test")]
    apply_events(inst, events)
    assert inst.dumps() == before


def test_arrival_delay_only_moves_the_target_vessel():
    inst = make_hard_instance("h", 8, 3, seed=1, slack=1.8)
    out = apply_events(inst, [Event("arrival_delay", 0, 2, 4, "test")])
    assert out.vessels[2].arrive == inst.vessels[2].arrive + 4
    for vid in range(inst.n):
        if vid != 2:
            assert out.vessels[vid].arrive == inst.vessels[vid].arrive


def test_events_beyond_the_cutoff_are_not_applied():
    inst = make_hard_instance("h", 8, 3, seed=1, slack=1.8)
    events = [Event("arrival_delay", 50, 0, 4, "late")]
    assert apply_events(inst, events, upto=10).vessels[0].arrive == \
        inst.vessels[0].arrive
    assert apply_events(inst, events, upto=60).vessels[0].arrive == \
        inst.vessels[0].arrive + 4


def test_crane_failure_recovers_after_its_window():
    inst = make_hard_instance("h", 8, 3, seed=1, slack=1.8)
    out = apply_events(inst, [Event("crane_loss", 0, None, 10, "test")])
    assert out.crane_failures
    start, end, reduction = out.crane_failures[0]
    assert reduction >= 1
    assert end > start


def test_observed_arrivals_reveal_the_past_only():
    planned = make_hard_instance("h", 8, 3, seed=1, slack=1.8)
    realised = apply_events(planned, [Event("arrival_delay", 0, 0, 6, "test")])
    seen = observed_arrivals(realised, planned, now=4)
    assert seen[0] == realised.vessels[0].arrive
    assert seen[1] == planned.vessels[1].arrive


def test_rolling_horizon_produces_a_feasible_realised_schedule():
    from bapbench.dynamics import rolling_horizon

    planned = make_hard_instance("roll", 10, 3, seed=20261014, slack=1.8)
    events = make_events(planned, random.Random(1000), n_arrival=1, n_overrun=1,
                         tidal_shift=True, crane_loss=True)
    realised = apply_events(planned, events)
    result = rolling_horizon(planned, realised, events, dt=8, time_limit=10.0)
    ok, _objective, violations = evaluate_schedule_on(realised, result.schedule)
    assert ok, violations


def test_rolling_horizon_regret_is_not_identically_zero():
    """Without decision commitment the metric collapses to zero."""
    from bapbench.dynamics import rolling_horizon

    planned = make_hard_instance("roll2", 10, 3, seed=20261014, slack=1.8)
    regrets = []
    for seed in range(4):
        events = make_events(planned, random.Random(3000 + seed), n_arrival=2,
                             n_overrun=1, tidal_shift=True, crane_loss=True)
        realised = apply_events(planned, events)
        perfect = BAPTimeIndexed(realised, time_limit=20.0).solve()
        result = rolling_horizon(planned, realised, events, dt=8, time_limit=10.0)
        ok, objective, _v = evaluate_schedule_on(realised, result.schedule)
        if ok and perfect.feasible and perfect.objective:
            regrets.append((objective - perfect.objective) / perfect.objective)
    assert regrets
    assert any(r > 0 for r in regrets)
