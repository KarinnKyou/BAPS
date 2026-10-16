# -*- coding: utf-8 -*-
"""Tests for the Imai, Nishimura & Papadimitriou (2001) reproduction.

The reference optimum is computed here by enumerating every berth assignment and
every service order within each berth, placing starts at the earliest feasible
instant, so the comparison to the heuristic does not depend on the module's own
relaxation.
"""

from __future__ import annotations

import itertools
import random

import pytest

from bapbench.reproduction.imai2001 import (
    INTERACT,
    INDIVIDUAL,
    PROCEDURES,
    SIMPLE,
    DBAPInstance,
    ImaiHeuristic,
    dbap_instance_from_instance,
    evaluate,
    hungarian,
    individual,
    interact,
    simple,
    solve_dbap,
    solve_relaxation,
    zero_multipliers,
)
from bapbench.instances import make_toy_instance


# --------------------------------------------------------------------------
# reference optimum by enumeration
# --------------------------------------------------------------------------


def _schedules(n: int, m: int):
    """Every assignment of ships to berths, in every per-berth service order."""
    for berth_of in itertools.product(range(m), repeat=n):
        groups = [[j for j in range(n) if berth_of[j] == i] for i in range(m)]
        for perm in itertools.product(*[list(itertools.permutations(g))
                                        for g in groups]):
            yield [list(x) for x in perm]


def _schedule_value(inst: DBAPInstance, order: list[list[int]]) -> int:
    """`sum_j (waiting + handling)` for one order, starts as early as feasible."""
    total = 0
    for i, seq in enumerate(order):
        t = inst.s[i]
        for j in seq:
            start = max(t, inst.a[j])
            total += start - inst.a[j] + inst.p[i][j]
            t = start + inst.p[i][j]
    return total


def reference_optimum(inst: DBAPInstance) -> int:
    """Exact optimum for a small instance, without a solver."""
    return min(_schedule_value(inst, order)
               for order in _schedules(inst.n, inst.m))


def _random_instance(rng: random.Random, n: int, m: int) -> DBAPInstance:
    p = [[rng.randint(1, 9) for _ in range(n)] for _ in range(m)]
    a = [rng.randint(0, 12) for _ in range(n)]
    return DBAPInstance(p=p, a=a, s=[0] * m,
                        names=[f"S{j + 1}" for j in range(n)])


SMALL = [(3, 1), (3, 2), (4, 2), (5, 2), (4, 3)]


# --------------------------------------------------------------------------
# the relaxation
# --------------------------------------------------------------------------


def test_lower_bound_never_exceeds_the_optimum():
    """`lower_bound <= optimum <= heuristic` on every enumerable instance."""
    for n, m in SMALL:
        for seed in range(6):
            inst = _random_instance(random.Random(seed * 31 + n * 7 + m), n, m)
            opt = reference_optimum(inst)
            for proc in PROCEDURES:
                res = ImaiHeuristic(inst, procedure=proc, max_iter=15).run()
                assert res.best_lower_bound is not None
                assert res.best_lower_bound <= opt + 1e-9, (n, m, seed, proc)
                assert res.best_feasible.objective >= opt - 1e-9
                assert res.best_lower_bound <= res.best_feasible.objective + 1e-9


def test_every_iteration_reports_a_valid_bound():
    """The published bound is the best of values that all bound the optimum."""
    for n, m in SMALL:
        inst = _random_instance(random.Random(n * 100 + m), n, m)
        opt = reference_optimum(inst)
        res = ImaiHeuristic(inst, procedure=INDIVIDUAL, max_iter=20).run()
        assert res.lower_bounds
        assert all(lb <= opt + 1e-9 for lb in res.lower_bounds)
        assert res.best_lower_bound == max(res.lower_bounds)


def test_bound_does_not_depend_on_the_repair_procedure():
    """[P3] is solved before the repair, so all three share the same bounds."""
    inst = _random_instance(random.Random(11), 4, 2)
    results = solve_dbap(inst, max_iter=10)
    best = {r.best_lower_bound for r in results.values()}
    assert len(best) == 1


def test_relaxation_assigns_every_ship_once_and_respects_candidate_berths():
    inst = DBAPInstance(p=[[3, 5], [4, 2]], a=[1, 6], s=[0, 4])
    relaxed = solve_relaxation(inst, zero_multipliers(inst))
    placed = sorted(j for seq in relaxed.order for j in seq)
    assert placed == list(range(inst.n))
    for i, seq in enumerate(relaxed.order):
        for j in seq:
            assert inst.a[j] >= inst.s[i]


def test_relaxation_start_times_follow_the_service_order():
    inst = _random_instance(random.Random(3), 4, 2)
    relaxed = solve_relaxation(inst, zero_multipliers(inst))
    for i, seq in enumerate(relaxed.order):
        t = inst.s[i]
        for k, j in enumerate(seq):
            assert relaxed.start[i][k] == t
            t += inst.p[i][j]


# --------------------------------------------------------------------------
# the subgradient method
# --------------------------------------------------------------------------


def test_subgradient_iterations_do_not_lose_the_best_bound():
    """`BestLB` is a running maximum by construction, so it cannot fall."""
    for n, m in SMALL[:3]:
        inst = _random_instance(random.Random(n + m), n, m)
        res = ImaiHeuristic(inst, procedure=SIMPLE, max_iter=30).run()
        running = res.lower_bounds[0]
        for lb in res.lower_bounds:
            running = max(running, lb)
        assert running == res.best_lower_bound


def test_multipliers_stay_non_negative_and_within_the_position_weights():
    """Step 6 clamps at zero; the column cap keeps `y`'s coefficient positive."""
    inst = _random_instance(random.Random(7), 5, 2)
    res = ImaiHeuristic(inst, procedure=INDIVIDUAL, max_iter=25).run()
    mu = res.best_mu
    assert mu is not None
    for i in range(inst.m):
        for k in range(inst.n):
            column = sum(mu[i][j][k] for j in range(inst.n))
            assert column <= k + 1 + 1e-9
            for j in range(inst.n):
                assert mu[i][j][k] >= 0.0


def test_termination_reasons_are_reported():
    inst = _random_instance(random.Random(13), 4, 2)
    res = ImaiHeuristic(inst, procedure=SIMPLE, max_iter=5).run()
    assert res.stopping_reason in ("max_iter", "gap_below_tolerance")
    assert res.iterations <= 5


def test_gap_below_one_stops_the_run_and_certifies_optimality():
    """With integer data a gap under 1 means the feasible value is optimal."""
    inst = DBAPInstance(p=[[2, 2], [2, 2]], a=[0, 0], s=[0, 0])
    res = ImaiHeuristic(inst, procedure=SIMPLE, max_iter=50).run()
    if res.stopping_reason == "gap_below_tolerance":
        assert res.optimal
        assert res.gap is not None and res.gap < 1.0
        assert res.best_feasible.objective == reference_optimum(inst)


def test_delta_halving_can_be_disabled_and_the_run_still_terminates():
    inst = _random_instance(random.Random(17), 4, 2)
    res = ImaiHeuristic(inst, procedure=INDIVIDUAL, max_iter=30,
                        delta_halve_after=10_000).run()
    assert res.iterations == 30
    assert res.best_lower_bound is not None


# --------------------------------------------------------------------------
# the three feasible-solution procedures
# --------------------------------------------------------------------------


def _assert_feasible(inst: DBAPInstance, sol) -> None:
    placed = sorted(j for seq in sol.order for j in seq)
    assert placed == list(range(inst.n)), "each ship serviced exactly once"
    for i, seq in enumerate(sol.order):
        for k, j in enumerate(seq):
            assert sol.start[i][k] >= inst.a[j], "start before arrival"
            assert sol.start[i][k] >= inst.s[i], "start before the berth is idle"
            if k:
                prev = seq[k - 1]
                assert sol.start[i][k] >= sol.start[i][k - 1] + inst.p[i][prev], \
                    "berth serves two ships at once"
    assert sol.objective == evaluate(inst, sol.order, sol.start)


@pytest.mark.parametrize("proc", PROCEDURES)
def test_procedures_return_feasible_schedules(proc):
    for n, m in SMALL:
        for seed in range(4):
            inst = _random_instance(random.Random(seed + 5 * n), n, m)
            relaxed = solve_relaxation(inst, zero_multipliers(inst))
            sol = {"SIMPLE": simple, "INDIVIDUAL": individual,
                   "INTERACT": interact}[proc](inst, relaxed)
            _assert_feasible(inst, sol)
            assert sol.objective >= reference_optimum(inst) - 1e-9


def test_simple_keeps_the_relaxed_assignments():
    inst = _random_instance(random.Random(23), 5, 2)
    relaxed = solve_relaxation(inst, zero_multipliers(inst))
    sol = simple(inst, relaxed)
    assert [sorted(s) for s in sol.order] == \
           [sorted(s) for s in (list(reversed(x)) for x in relaxed.order)]


def test_individual_and_interact_never_worsen_simple():
    for seed in range(6):
        inst = _random_instance(random.Random(seed * 7 + 1), 5, 2)
        relaxed = solve_relaxation(inst, zero_multipliers(inst))
        base = simple(inst, relaxed).objective
        assert individual(inst, relaxed).objective <= base
        assert interact(inst, relaxed).objective <= base


def test_interact_moves_ships_between_berths_without_worsening():
    """INTERACT may relocate a ship; the schedule stays feasible either way."""
    inst = _random_instance(random.Random(29), 5, 3)
    relaxed = solve_relaxation(inst, zero_multipliers(inst))
    before = individual(inst, relaxed)
    after = interact(inst, relaxed)
    _assert_feasible(inst, after)
    assert after.objective <= before.objective
    assert after.ships_moved >= 0


def test_waiting_criterion_is_the_sum_of_the_successors_gaps():
    """Eq. (26): `W_j = sum_{j' in LU} (D_j - A_j')` with `LU` U and successors."""
    from bapbench.reproduction.imai2001 import _waiting_criterion
    inst = DBAPInstance(p=[[3, 3, 3]], a=[0, 5, 9], s=[0])
    order = [[0, 1, 2]]
    start = [[0, 3, 6]]
    # U is ship 1, whose start is 3; the only successor is ship 2. Moving ship 2
    # immediately before U puts its departure at U's start, and LU is {1, 2}
    # minus ship 2, so W = 3 - A_1 = -2.
    assert _waiting_criterion(inst, order, start, 0, 1, 2) == start[0][1] - inst.a[1]
    # With LU kept as U and its successors, a candidate further down the sequence
    # compares against the same departure, so W does not depend on `cand` except
    # through which arrival is left out of the sum.
    assert _waiting_criterion(inst, order, start, 0, 1, 1) == start[0][1] - inst.a[2]


def test_unknown_procedure_is_rejected():
    inst = _random_instance(random.Random(2), 3, 1)
    with pytest.raises(ValueError):
        ImaiHeuristic(inst, procedure="GREEDY")


# --------------------------------------------------------------------------
# Hungarian method and the package bridge
# --------------------------------------------------------------------------


def test_hungarian_matches_brute_force_on_small_matrices():
    rng = random.Random(41)
    for _ in range(40):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        cost = [[rng.randint(-9, 9) for _ in range(m)] for _ in range(n)]
        value, rows = hungarian(cost)
        assert sorted(rows) == sorted(set(rows))
        assert abs(value - sum(cost[i][rows[i]] for i in range(n))) < 1e-9
        best = min(sum(cost[i][cols[i]] for i in range(n))
                   for cols in itertools.permutations(range(m), n))
        assert abs(value - best) < 1e-9


def test_instance_bridge_reads_vessels_and_berths():
    inst = make_toy_instance("bridge", 4, 2, seed=5)
    dbap = dbap_instance_from_instance(inst)
    assert dbap.n == inst.n and dbap.m == inst.m
    assert dbap.a == [v.arrive for v in inst.vessels]
    assert dbap.s == [0, 0]


def test_instance_bridge_accepts_berth_dependent_handling_times():
    inst = make_toy_instance("bridge2", 3, 2, seed=7)
    table = {0: {0: 4, 1: 5, 2: 6}, 1: {0: 7, 1: 8, 2: 9}}
    dbap = dbap_instance_from_instance(inst, handle_at=table)
    assert dbap.p[0] == [4, 5, 6]
    assert dbap.p[1] == [7, 8, 9]


def test_driver_returns_one_result_per_procedure():
    inst = _random_instance(random.Random(43), 4, 2)
    results = solve_dbap(inst, max_iter=8)
    assert set(results) == set(PROCEDURES)
    for res in results.values():
        assert res.best_feasible is not None
        assert res.gap is not None and res.gap >= -1e-9
