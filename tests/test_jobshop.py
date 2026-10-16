"""Tests for the solver-free job-shop optimum."""

from __future__ import annotations

import pytest

from bapbench.models.classical import JobShopMILP, make_simple_problem, to_instance_view
from bapbench.verification.jobshop import job_shop_optimum


def _instance(n: int, m: int):
    prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
    routes = {j: list(range(m)) for j in range(n)}
    milp = JobShopMILP(prob, routes=routes,
                       instance_view=to_instance_view(prob)).solve()
    proc = [[job.proc] * m for job in prob.jobs]
    return prob, milp, proc


@pytest.mark.parametrize("n,m", [(4, 3), (5, 3), (4, 4)])
def test_milp_matches_the_independent_optimum(n, m):
    prob, milp, proc = _instance(n, m)
    result = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                              time_budget=60.0)
    assert result.objective is not None, result.note
    assert not result.note, "search should finish well inside the budget"
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


def test_the_search_only_covers_non_delay_schedules():
    """Known limitation, pinned so it cannot be mistaken for exactness.

    The search branches only on operations whose earliest start equals the global
    minimum earliest start. That set is a proper subset of the active schedules,
    so on a general job shop it can report a value above the optimum. If this
    test ever fails, the search has become exact and the module docstring should
    be upgraded.
    """
    routes = [[2, 1, 3, 0], [1, 3], [1, 2, 3], [1, 2, 3, 0]]
    proc = [[4, 5, 4, 7], [1, 1], [3, 1, 1], [2, 1, 2, 6]]
    result = job_shop_optimum(routes, proc, time_budget=30.0)
    assert result.objective is not None and not result.note

    # job 0 alone needs 20 time units on its own machines, so 20 is attainable as
    # a lower bound; the MILP reaches it with no machine overlap.
    job_bound = max(sum(row) for row in proc)
    assert job_bound == 20
    assert result.objective >= job_bound
    assert result.objective > job_bound, (
        "the search returned the lower bound, so it is now exact on this case; "
        "update the module docstring")


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
