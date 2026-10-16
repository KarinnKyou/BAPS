"""Exhaustive optimum for small instances, computed without a solver.

Independent of the MILP, so agreement between the two is real evidence.

The enumeration must respect every constraint the exact model enforces, or the
two are solving different problems and agreement is meaningless. In particular a
schedule is only valid if it keeps vessels clear of berth closures and keeps the
number of vessels handled at once within the crane limit. The crane limit couples
the berths, so start times cannot be chosen berth by berth; the search below
assigns start times in a fixed vessel order and backtracks when a berth cannot
take the vessel at or after its arrival.

The search space is exponential, so this is limited to small instances. When it
cannot finish - either because the space is too large or because the crane limit
rules everything out - the result reports that rather than returning a bound.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass

from ..instances import Instance, Schedule


@dataclass
class ExhaustiveResult:
    objective: float | None
    assign: dict[int, tuple[int, int]] | None
    n_cases: int
    wall_time: float
    feasible: bool = True
    note: str = ""


def _blocked(inst: Instance, bid: int, start: int, handle: int) -> bool:
    return any(inst.blocked(bid, t) for t in range(start, start + handle))


def _feasible_starts(inst: Instance, bid: int, arrive: int, handle: int) -> list[int]:
    last = inst.horizon - handle
    return [t for t in range(arrive, last + 1)
            if not _blocked(inst, bid, t, handle)]


def _search(inst: Instance, berth_of: dict[int, int], order: list[int],
            deadline: float) -> tuple[dict[int, tuple[int, int]] | None, int, bool]:
    """Assign start times in `order`, returning (assignment, cases, timed_out)."""
    assign: dict[int, tuple[int, int]] = {}
    busy = [0] * inst.horizon
    cases = 0
    limit = inst.quay_crane_limit

    def rec(idx: int) -> dict[int, tuple[int, int]] | None:
        nonlocal cases
        if time.perf_counter() > deadline:
            raise TimeoutError
        if idx == len(order):
            return dict(assign)
        vid = order[idx]
        bid = berth_of[vid]
        v = inst.vessels[vid]
        for start in _feasible_starts(inst, bid, v.arrive, v.handle):
            span = range(start, start + v.handle)
            if any(busy[t] for t in span):
                continue
            if limit is not None and any(
                    (inst.crane_capacity(t) or 0) <= busy[t] for t in span):
                continue
            for t in span:
                busy[t] += 1
            assign[vid] = (bid, start)
            cases += 1
            found = rec(idx + 1)
            if found is not None:
                return found
            del assign[vid]
            for t in span:
                busy[t] -= 1
        return None

    try:
        return rec(0), cases, False
    except TimeoutError:
        return None, cases, True


def exhaustive_optimum(inst: Instance, objective: str = "sum_C",
                       max_cases: int = 5_000_000,
                       time_budget: float = 120.0) -> ExhaustiveResult:
    """Return the optimum by enumeration, or report why it could not be computed."""
    n, m = inst.n, inst.m
    t0 = time.perf_counter()
    deadline = t0 + time_budget

    if inst.quay_crane_limit is not None and inst.min_crane_capacity() is None:
        return ExhaustiveResult(None, None, 0, 0.0, feasible=False,
                                note="no crane capacity is ever available")

    est = (m ** n) * max(1, _perm_upper(n, m))
    if est > max_cases:
        return ExhaustiveResult(None, None, 0, 0.0, feasible=False,
                                note=f"search space estimate {est:.3e} exceeds budget "
                                     f"{max_cases:.1e}; n={n}, m={m} too large to "
                                     f"enumerate")

    best_obj: float | None = None
    best_assign: dict[int, tuple[int, int]] | None = None
    cases = 0
    timed_out = False

    # The objective depends only on start times, not on the visiting order within
    # a berth, so one order per assignment is enough: ships cannot overtake on a
    # single berth without one of them starting later than necessary.
    for combo in itertools.product(range(m), repeat=n):
        berth_of = {i: combo[i] for i in range(n)}
        order = sorted(range(n), key=lambda i: (inst.vessels[i].arrive,
                                                -inst.vessels[i].handle))
        found, sub_cases, over = _search(inst, berth_of, order, deadline)
        cases += sub_cases
        if over:
            timed_out = True
            break
        if found is None:
            continue
        comp = {i: found[i][1] + inst.vessels[i].handle for i in range(n)}
        obj = float(sum(comp.values()) if objective == "sum_C" else max(comp.values()))
        if best_obj is None or obj < best_obj:
            best_obj, best_assign = obj, found

    wall = time.perf_counter() - t0
    if timed_out:
        return ExhaustiveResult(best_obj, best_assign, cases, wall, feasible=False,
                                note=f"exceeded the {time_budget}s budget after "
                                     f"{cases} expansions; may not be optimal")
    if best_obj is None:
        return ExhaustiveResult(None, None, cases, wall, feasible=False,
                                note="no feasible schedule found")
    return ExhaustiveResult(best_obj, best_assign, cases, wall)


def _perm_upper(n: int, m: int) -> float:
    """Order-of-magnitude bound used only to decide whether to attempt a search."""
    import math
    return math.factorial(n)


def _capacity_ok(inst: Instance, assign: dict[int, tuple[int, int]]) -> bool:
    """Whole-berth occupancy, implemented apart from Schedule.check.

    A vessel occupies its entire berth while handling, so two vessels may never
    hold the same berth in the same period.
    """
    by_berth: dict[int, list[tuple[int, int]]] = {}
    for i, (bid, st) in assign.items():
        by_berth.setdefault(bid, []).append((st, st + inst.vessels[i].handle))
    for spans in by_berth.values():
        spans.sort()
        for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            if s2 < e1:
                return False
    return True


def verify_instance(inst: Instance, milp_objective: float | None,
                    objective: str = "sum_C",
                    max_cases: int = 5_000_000,
                    time_budget: float = 120.0) -> tuple[bool, ExhaustiveResult]:
    """Compare the MILP objective with the exhaustive optimum."""
    res = exhaustive_optimum(inst, objective, max_cases, time_budget)
    if res.objective is None or milp_objective is None:
        return False, res
    return abs(res.objective - milp_objective) < 1e-6, res


if __name__ == "__main__":
    from ..instances import make_toy_instance

    inst = make_toy_instance("enum_demo", 5, 3, seed=20261014)
    result = exhaustive_optimum(inst, max_cases=2_000_000)
    print(f"optimum {result.objective} over {result.n_cases} expansions "
          f"in {result.wall_time:.3f}s")
    sys.exit(0 if result.objective is not None else 1)
