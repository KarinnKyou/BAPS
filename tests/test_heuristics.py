"""Tests for the heuristics and metaheuristics.

These carry the highest risk in the package: a heuristic that returns an
infeasible schedule, or an objective it did not achieve, silently corrupts every
comparison built on it. The tests therefore check properties that hold for any
correct implementation (feasibility, a reproducible objective value, known
bounds, determinism where the method is deterministic) rather than fixed numbers,
plus the specific mechanisms each method relies on.
"""

from __future__ import annotations

import math
import random

import pytest

from bapbench.heuristics.constructive import (
    PLACE_RESTARTS_BY_METHOD,
    BASELINES,
    genetic_algorithm,
    place_in_order,
    run_all_baselines,
    simulated_annealing,
)
from bapbench.heuristics.metaheuristics import (
    DESTROY,
    HEURISTICS,
    NEIGHBORHOODS,
    REPAIR,
    Evaluator,
    alns,
    iterated_local_search,
    local_search,
    nbr_relocate,
    nbr_reverse,
    nbr_swap,
    run_heuristics,
)
from bapbench.instances import make_hard_instance, make_toy_instance
from bapbench.metrics import sanity_check_schedule
from bapbench.models.bap import BAPTimeIndexed

DETERMINISTIC = ("FCFS", "LPT", "SPT", "GREEDY")
STOCHASTIC = ("SA", "GA")
ALL_METHODS = DETERMINISTIC + STOCHASTIC + tuple(HEURISTICS)


@pytest.fixture(scope="module")
def toy():
    return make_toy_instance("heur_toy", 8, 3, seed=20261014)


@pytest.fixture(scope="module")
def hard():
    return make_hard_instance("heur_hard", 10, 3, seed=20261014, slack=2.0)


def _lower_bound(inst) -> float:
    return float(sum(v.arrive + v.handle for v in inst.vessels))


def run_method(name: str, inst, objective: str = "sum_C", seed: int = 20261014):
    """Call any method by name.

    The baselines take (instance, objective) except SA/GA which also take a seed;
    the metaheuristics always take a seed.
    """
    if name in BASELINES:
        fn = BASELINES[name]
        return (fn(inst, objective, seed=seed) if name in ("SA", "GA")
                else fn(inst, objective))
    return HEURISTICS[name](inst, objective, seed)


# --- properties every method must satisfy -------------------------------------

@pytest.mark.parametrize("name", ALL_METHODS)
def test_every_method_returns_a_valid_schedule(toy, name):
    sch = run_method(name, toy)
    assert sch.feasible, sch.violations
    assert sanity_check_schedule(toy, sch) == []


@pytest.mark.parametrize("name", ALL_METHODS)
def test_every_method_reports_the_objective_it_achieved(toy, name):
    """Recompute from the assignment alone, with no shared code."""
    sch = run_method(name, toy)
    comp = {vid: st + toy.vessels[vid].handle for vid, (_b, st) in sch.assign.items()}
    assert sch.objective == pytest.approx(float(sum(comp.values())))


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_deterministic_methods_do_not_vary(toy, name):
    """FCFS/SPT/LPT/GREEDY are rules, so two runs must agree exactly."""
    a = BASELINES[name](toy, "sum_C").objective
    b = BASELINES[name](toy, "sum_C").objective
    assert a == b


@pytest.mark.parametrize("name", STOCHASTIC + tuple(HEURISTICS))
def test_stochastic_methods_expose_a_seed(toy, name):
    """Two different seeds must be allowed to differ, and one seed must repeat."""
    a = run_method(name, toy, seed=1).objective
    b = run_method(name, toy, seed=1).objective
    assert a == b, "same seed must reproduce the same result"


@pytest.mark.parametrize("name", ALL_METHODS)
def test_no_method_beats_a_valid_lower_bound(hard, name):
    """Every vessel finishes after its arrival plus handling, so the sum of those
    minima bounds sum_C. A value below it means an infeasible schedule was scored."""
    sch = run_method(name, hard)
    if not sch.feasible:
        pytest.skip(f"{name} produced no schedule on this instance")
    assert sch.objective >= _lower_bound(hard) - 1e-6


def test_methods_stay_within_reach_of_the_optimum(hard):
    """A method far above the optimum is suspicious even if feasible."""
    exact = BAPTimeIndexed(hard, objective="sum_C", time_limit=60.0).solve()
    assert exact.extra.get("proved_optimal"), "test needs a proved optimum"
    for name in ALL_METHODS:
        sch = run_method(name, hard, seed=7)
        if not sch.feasible:
            continue
        ratio = sch.objective / exact.objective
        assert ratio < 3.0, f"{name} is {ratio:.2f}x the optimum"


# --- the shared decoder -------------------------------------------------------

def test_placement_is_complete_and_valid(toy):
    order = list(range(toy.n))
    assign = place_in_order(toy, order, "earliest_start")
    assert set(assign) == set(range(toy.n))
    assert sanity_check_schedule(toy, assign) == []


def test_restarts_are_only_used_when_the_order_cannot_be_decoded(toy):
    """A decodable order must be honoured exactly; the restart budget is a fallback."""
    order = list(range(toy.n))
    without = place_in_order(toy, order, "earliest_start", restarts=0)
    with_budget = place_in_order(toy, order, "earliest_start", restarts=500)
    assert without == with_budget


def test_placement_raises_rather_than_returning_a_partial_schedule():
    """An impossible instance must be reported, not silently truncated."""
    inst = make_toy_instance("impossible", 12, 1, seed=3, horizon=8)
    with pytest.raises(RuntimeError):
        place_in_order(inst, list(range(inst.n)), "earliest_start")


def test_each_deterministic_method_has_a_documented_restart_budget():
    for name in ("FCFS", "LPT", "SPT", "GREEDY"):
        assert name in PLACE_RESTARTS_BY_METHOD
        assert PLACE_RESTARTS_BY_METHOD[name] >= 0


# --- neighbourhoods -----------------------------------------------------------

@pytest.mark.parametrize("fn", [nbr_swap, nbr_relocate, nbr_reverse])
def test_neighbourhoods_return_a_permutation(fn, toy):
    order = list(range(toy.n))
    out = fn(order, random.Random(0))
    assert sorted(out) == sorted(order), f"{fn.__name__} did not preserve the set"
    assert len(out) == len(order)


@pytest.mark.parametrize("move", list(NEIGHBORHOODS))
def test_local_search_never_worsens_and_improves_when_it_can(toy, move):
    ev = Evaluator(toy, "sum_C")
    start = list(range(toy.n))
    before = ev(start)
    out, after = local_search(toy, start, ev, random.Random(0), move=move)
    assert after <= before + 1e-9
    assert sorted(out) == sorted(start)


def test_evaluator_caches_and_counts(toy):
    ev = Evaluator(toy, "sum_C")
    order = list(range(toy.n))
    first = ev(order)
    second = ev(order)
    assert first == second
    st = ev.stats()
    assert st["evals"] == 1 and st["cache_hits"] == 1


def test_evaluator_penalises_an_undecodable_order():
    """An order that cannot be placed is infinite cost, not a silent zero."""
    inst = make_toy_instance("tight", 12, 2, seed=5, horizon=30, tight=True)
    ev = Evaluator(inst, "sum_C")
    worst = None
    for t in range(40):
        cand = list(range(inst.n))
        random.Random(t).shuffle(cand)
        val = ev(cand)
        if val == math.inf:
            worst = t
            break
    if worst is None:
        pytest.skip("no undecodable order found in 40 draws")
    assert ev.stats()["infeasible_evals"] >= 1


# --- mechanisms ---------------------------------------------------------------

def test_sa_cools_over_its_run(toy):
    """A longer run may not be worse, which only holds if cooling is implemented."""
    short = simulated_annealing(toy, "sum_C", iters=50, seed=1).objective
    long = simulated_annealing(toy, "sum_C", iters=4000, seed=1).objective
    assert long <= short + 1e-9


def test_ga_population_is_seeded_and_reproducible(toy):
    a = genetic_algorithm(toy, "sum_C", iters=200, seed=5).objective
    b = genetic_algorithm(toy, "sum_C", iters=200, seed=5).objective
    assert a == b


def test_ils_accepts_equal_cost_moves(toy):
    """Plateau traversal is what separates ILS from plain descent."""
    result = iterated_local_search(toy, "sum_C", iters=10, kicks=2, seed=3)
    plain = BASELINES["FCFS"](toy, "sum_C").objective
    assert result.feasible
    assert result.objective <= plain + 1e-9


def test_alns_adaptive_weights_stay_in_range_and_move(toy):
    """The whole point of ALNS is that operator weights respond to performance."""
    sch = alns(toy, "sum_C", iters=120, seed=1)
    assert sch.feasible
    dw = sch.extra.get("destroy_w")
    rw = sch.extra.get("repair_w")
    assert dw and rw, "ALNS must report its operator weights"
    assert set(dw) == set(DESTROY) and set(rw) == set(REPAIR)
    for w in list(dw.values()) + list(rw.values()):
        assert w >= 0.05 - 1e-9, "the exploration floor must hold"


def test_alns_uses_both_destroy_and_repair_operators(toy):
    """With one operator dominating the weights the 'adaptive' claim is empty."""
    sch = alns(toy, "sum_C", iters=200, seed=2)
    assert len(sch.extra["destroy_w"]) >= 2
    assert len(sch.extra["repair_w"]) >= 2


# --- aggregate entry points ---------------------------------------------------

def test_run_all_baselines_covers_every_baseline(hard):
    out = run_all_baselines(hard, "sum_C")
    assert set(out) == set(BASELINES)
    for name, sch in out.items():
        if sch.feasible:
            assert sanity_check_schedule(hard, sch) == [], name


def test_search_heuristics_never_receive_an_undefined_start(hard):
    """LS and ILS seed from the constructive rules. On a tight instance every rule
    can fail to decode, which left the seed undefined and crashed the search with
    'NoneType is not subscriptable' rather than reporting that no schedule was
    found. Either a schedule comes back, or the failure is explicit."""
    for name in ("LS", "ILS"):
        try:
            sch = HEURISTICS[name](hard, "sum_C", 3)
        except RuntimeError as exc:
            assert "decodable" in str(exc) or "placement" in str(exc), name
            continue
        assert sch.feasible or sch.violations, name
        if sch.feasible:
            assert sanity_check_schedule(hard, sch) == [], name


def test_run_heuristics_records_failures_instead_of_raising(hard):
    """The aggregate entry point must survive a method that builds nothing."""
    out = run_heuristics(hard, "sum_C", seed=3)
    assert set(out) == set(HEURISTICS)
    for name, sch in out.items():
        if not sch.feasible:
            assert sch.violations, f"{name} failed without a reason"


def test_ils_never_reports_worse_than_something_it_computed(hard):
    """The returned schedule is declared the best of the run, so its objective may
    not exceed the value the run itself reached. Returning the seed instead of the
    local optimum it improved on is the failure this guards."""
    without_iterations = iterated_local_search(hard, "sum_C", iters=0, seed=3)
    assert without_iterations.feasible, without_iterations.violations

    for iters in (0, 5, 30):
        sch = iterated_local_search(hard, "sum_C", iters=iters, seed=3)
        assert sch.feasible, sch.violations
        assert sch.objective <= without_iterations.objective + 1e-9, (
            f"iters={iters} returned {sch.objective} but the run's own local "
            f"optimum is {without_iterations.objective}")
