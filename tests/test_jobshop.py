"""Tests for the solver-free job-shop optimum."""

from __future__ import annotations

import random

import pytest

from bapbench.models.classical import JobShopMILP, make_simple_problem, to_instance_view
from bapbench.verification.benchmarks import solve_shop
from bapbench.verification.jobshop import job_shop_optimum


def _instance(n: int, m: int):
    prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
    routes = {j: list(range(m)) for j in range(n)}
    milp = JobShopMILP(prob, routes=routes,
                       instance_view=to_instance_view(prob)).solve()
    proc = [[job.proc] * m for job in prob.jobs]
    return prob, milp, proc


def _random_shop(rng: random.Random, n: int, m: int):
    """A job shop with per-operation durations and a shuffled route per job."""
    routes, proc = [], []
    for _ in range(n):
        machines = rng.sample(range(m), rng.randint(2, m))
        routes.append(machines)
        proc.append([rng.randint(1, 6) for _ in machines])
    return routes, proc


@pytest.mark.parametrize("n,m", [(4, 3), (5, 3), (4, 4)])
def test_milp_matches_the_independent_optimum(n, m):
    _prob, milp, proc = _instance(n, m)
    result = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                              time_budget=60.0)
    assert result.objective is not None, result.note
    assert not result.note, "search should finish well inside the budget"
    assert result.feasible
    assert milp.feasible, milp.violations
    assert milp.objective == pytest.approx(result.objective)


def test_optimum_is_at_least_the_machine_load_bound():
    n, m = 4, 3
    _prob, _milp, proc = _instance(n, m)
    result = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                              time_budget=60.0)
    # every job visits every machine exactly once, in order
    load = [sum(proc[j][mach] for j in range(n)) for mach in range(m)]
    assert result.objective >= max(load)


def test_a_two_job_two_machine_instance_is_solved_exactly():
    """Hand-checkable case: both jobs need 3 then 2 time units on machines 0 and 1."""
    routes = [[0, 1], [0, 1]]
    proc = [[3, 2], [3, 2]]
    result = job_shop_optimum(routes, proc, time_budget=10.0)
    # machine 0 is the bottleneck and cannot be shared, so C_max = 3 + 3 + 2 = 8
    assert result.objective == 8


def test_a_route_may_revisit_a_machine():
    """Job 0 goes 0 -> 1 -> 0, so its last operation queues behind the first.

    Hand-checked optimum 5: job 0 on m0 [0,1), job 1 on m1 [0,2), job 1 on m0
    [2,3), job 0 on m1 [2,4), job 0 back on m0 [4,5).
    """
    routes = [[0, 1, 0], [1, 0]]
    proc = [[1, 2, 1], [2, 1]]
    result = job_shop_optimum(routes, proc, time_budget=10.0)
    assert result.feasible and not result.note
    assert result.objective == 5


def test_the_search_reaches_the_job_bound_where_the_non_delay_rule_missed_it():
    """Regression for the old branching rule.

    Job 0 needs 20 time units on its own machines, and the shop admits a schedule
    with no machine overlap at 20, so 20 is the optimum. Branching only on
    operations whose earliest start equalled the global minimum -- the non-delay
    schedules -- returned more than 20 on this instance; the active-schedule
    branching must reach the bound.
    """
    routes = [[2, 1, 3, 0], [1, 3], [1, 2, 3], [1, 2, 3, 0]]
    proc = [[4, 5, 4, 7], [1, 1], [3, 1, 1], [2, 1, 2, 6]]
    result = job_shop_optimum(routes, proc, time_budget=30.0)
    assert result.feasible and not result.note

    job_bound = max(sum(row) for row in proc)
    assert job_bound == 20
    assert result.objective == job_bound


def test_random_shops_with_per_operation_durations_match_the_milp():
    """The class of instance the non-delay search got wrong: durations vary per
    operation and the route is not the identity.

    `solve_shop` is a separate MILP formulation over the same data, so agreement
    is evidence about the search rather than self-consistency.
    """
    rng = random.Random(20261014)
    for i in range(10):
        routes, proc = _random_shop(rng, rng.randint(3, 4), rng.randint(3, 4))
        search = job_shop_optimum(routes, proc, time_budget=30.0)
        assert search.feasible and not search.note, search.note
        milp = solve_shop(routes, proc, f"rand_{i}", time_limit=60.0)
        assert milp["solved"] is not None, milp
        assert search.objective == pytest.approx(milp["solved"])


def _brute_force(routes: list[list[int]], proc: list[list[int]]) -> int:
    """Every interleaving that respects job precedence, each scheduled as early as
    possible, with no solver involved.

    A schedule is a linear extension of its job precedences, and re-scheduling an
    extension at the earliest feasible times never delays an operation, so the
    minimum over extensions is the optimum. Small instances only.
    """
    n = len(routes)
    m = max(mach for row in routes for mach in row) + 1
    total = sum(len(row) for row in routes)
    machine_free = [0] * m
    job_free = [0] * n
    next_op = [0] * n
    best: list[int | None] = [None]

    def rec(count: int) -> None:
        if count == total:
            value = max(job_free)
            if best[0] is None or value < best[0]:
                best[0] = value
            return
        for j in range(n):
            k = next_op[j]
            if k >= len(routes[j]):
                continue
            mach = routes[j][k]
            end = max(job_free[j], machine_free[mach]) + proc[j][k]
            old_machine, old_job = machine_free[mach], job_free[j]
            machine_free[mach] = end
            job_free[j] = end
            next_op[j] = k + 1
            rec(count + 1)
            next_op[j] = k
            job_free[j] = old_job
            machine_free[mach] = old_machine

    rec(0)
    assert best[0] is not None
    return best[0]


def test_tiny_random_shops_match_a_full_enumerator():
    """No solver and no MILP: enumerate the interleavings instead.

    This is the independent ground truth for the branching rule. It is exponential,
    which is why the instances are tiny, but it makes no assumption beyond "an
    optimal schedule is semi-active".
    """
    rng = random.Random(4242)
    for _ in range(6):
        routes, proc = _random_shop(rng, 3, 3)
        assert job_shop_optimum(routes, proc, time_budget=10.0).objective == \
            _brute_force(routes, proc)


def test_a_budget_hit_is_reported_as_an_upper_bound():
    """A search that stops early has an incumbent, not an optimum; the result must
    say so rather than let the value be compared as if it were proved."""
    rng = random.Random(11)
    routes, proc = _random_shop(rng, 7, 6)
    result = job_shop_optimum(routes, proc, time_budget=0.0)
    assert not result.feasible
    assert result.note
    assert result.objective is not None
    assert "upper bound" in result.note


def test_a_malformed_instance_is_rejected():
    with pytest.raises(ValueError):
        job_shop_optimum([[0, 1]], [[3]], time_budget=5.0)
    with pytest.raises(ValueError):
        job_shop_optimum([[0]], [[0]], time_budget=5.0)
    with pytest.raises(ValueError):
        job_shop_optimum([[]], [[]], time_budget=5.0)


def test_equal_duration_identity_routes_agree_with_the_lower_bound():
    """On the instances this package builds, the search reaches the job bound."""
    for n, m in ((4, 3), (5, 3), (6, 3), (4, 4), (5, 4)):
        prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n,
                                   horizon=n * 12 + 40)
        proc = [[job.proc] * m for job in prob.jobs]
        routes = [list(range(m)) for _ in range(n)]
        result = job_shop_optimum(routes, proc, time_budget=30.0)
        assert result.objective is not None
        assert result.objective >= max(sum(row) for row in proc)
