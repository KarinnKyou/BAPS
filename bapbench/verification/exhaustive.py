"""Exhaustive optimum for small instances, computed without a solver.

Independent of the MILP, so agreement between the two is real evidence.

The enumeration must respect every constraint the exact model enforces, or the
two are solving different problems and agreement is meaningless. A schedule is
only valid if it keeps vessels clear of berth closures and keeps the number of
vessels handled at once within the crane limit. The crane limit couples the
berths, so start times cannot be chosen berth by berth.

For a fixed berth assignment the search below is exact: it places start times
vessel by vessel in a fixed order and tries every feasible start in the
vessel's window, backtracking on failure. Occupancy per berth and the crane
count per period are the running state. The candidate set is finite, so
backtracking over it covers every schedule the assignment admits; the vessel
order cannot lose a schedule because each vessel is offered its whole candidate
window. A branch is cut only by a completion-time lower bound that cannot beat
the best objective already found.

The search space is exponential, so this is limited to small instances. When it
cannot finish - because the space is too large, or because the time or case
budget trips - the result reports that rather than returning a bound.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from dataclasses import dataclass

from ..instances import Instance


@dataclass
class ExhaustiveResult:
    objective: float | None
    assign: dict[int, tuple[int, int]] | None
    n_cases: int
    wall_time: float
    feasible: bool = True
    note: str = ""


@dataclass
class _Best:
    """Incumbent carried across assignments."""
    obj: float
    assign: dict[int, tuple[int, int]] | None
    cases: int = 0
    truncated: bool = False


def _blocked(inst: Instance, bid: int, start: int, handle: int) -> bool:
    return any(inst.blocked(bid, t) for t in range(start, start + handle))


def _window(inst: Instance, bid: int, arrive: int, handle: int) -> list[int]:
    """Start periods that fit the horizon and avoid the berth's closures."""
    last = inst.horizon - handle
    return [t for t in range(arrive, last + 1)
            if not _blocked(inst, bid, t, handle)]


def _perm_upper(n: int, m: int) -> float:
    """Order-of-magnitude bound used only to decide whether to attempt a search."""
    return math.factorial(n)


def _search_assignment(inst: Instance, berth_of: dict[int, int], order: list[int],
                       objective: str, best: _Best, deadline: float,
                       max_cases: int) -> None:
    """Exact start-time search for one berth assignment, updating `best`.

    `order` lists the vessels; `berth[i]` is the berth the i-th vessel is fixed
    to. Each vessel is offered every start in its window, and a start is rejected
    when a period it needs is occupied on that berth, or when the crane count is
    already at capacity.
    """
    n = len(order)
    handle = [inst.vessels[vid].handle for vid in order]
    span_of = [
        [(st, range(st, st + handle[i]))
         for st in _window(inst, berth_of[vid], inst.vessels[vid].arrive,
                           handle[i])]
        for i, vid in enumerate(order)
    ]
    berth = [berth_of[vid] for vid in order]
    comp_min = [inst.vessels[vid].arrive + handle[i] for i, vid in enumerate(order)]

    # Completion the vessels from position i on could reach if nothing hindered
    # them: a sum for total completion time, a maximum for the makespan.
    rest = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        if objective == "sum_C":
            rest[i] = rest[i + 1] + comp_min[i]
        else:
            rest[i] = max(rest[i + 1], float(comp_min[i]))

    limit = inst.quay_crane_limit
    cap = inst.crane_capacity
    # Whole-berth occupancy is per berth: a vessel on one berth blocks only that
    # berth. The crane count is terminal-wide, so it is one shared array.
    berth_busy = [[0] * inst.horizon for _ in range(inst.m)]
    crane_busy = [0] * inst.horizon
    placement = [0] * n
    cases = 0

    def rec(i: int, partial: float) -> None:
        nonlocal cases
        if i == n:
            if best.assign is None or partial < best.obj:
                best.obj = partial
                best.assign = {order[j]: (berth[j], placement[j])
                               for j in range(n)}
            return
        for start, span in span_of[i]:
            if cases >= max_cases:
                best.truncated = True
                return
            if time.perf_counter() > deadline:
                best.truncated = True
                return
            cases += 1
            if objective == "sum_C":
                bound = partial + start + handle[i] + rest[i + 1]
            else:
                bound = max(partial, start + handle[i], rest[i + 1])
            if best.assign is not None and bound >= best.obj:
                continue
            if any(berth_busy[berth[i]][t] for t in span):
                continue
            if limit is not None and any(crane_busy[t] >= (cap(t) or 0)
                                         for t in span):
                continue
            for t in span:
                berth_busy[berth[i]][t] = 1
                crane_busy[t] += 1
            placement[i] = start
            rec(i + 1, partial + start + handle[i] if objective == "sum_C"
                else max(partial, start + handle[i]))
            for t in span:
                berth_busy[berth[i]][t] = 0
                crane_busy[t] -= 1
            if best.truncated:
                return

    rec(0, 0.0)
    best.cases += cases


def exhaustive_optimum(inst: Instance, objective: str = "sum_C",
                       max_cases: int = 5_000_000,
                       time_budget: float = 120.0) -> ExhaustiveResult:
    """Return the optimum by enumeration, or report why it could not be computed."""
    if objective not in ("sum_C", "C_max"):
        raise ValueError(f"unknown objective: {objective}")

    n, m = inst.n, inst.m
    t0 = time.perf_counter()

    if inst.quay_crane_limit is not None and inst.min_crane_capacity() is None:
        return ExhaustiveResult(None, None, 0, 0.0, feasible=False,
                                note="no crane capacity is ever available")

    est = (m ** n) * max(1, _perm_upper(n, m))
    if est > max_cases:
        return ExhaustiveResult(None, None, 0, 0.0, feasible=False,
                                note=f"search space estimate {est:.3e} exceeds budget "
                                     f"{max_cases:.1e}; n={n}, m={m} too large to "
                                     f"enumerate")

    # Any vessel order is complete here: each vessel gets its whole candidate
    # window, so ordering only affects how fast a good incumbent appears.
    order = sorted(range(n), key=lambda i: (-inst.vessels[i].handle,
                                            inst.vessels[i].arrive, i))

    deadline = t0 + time_budget
    best = _Best(obj=math.inf, assign=None)
    for combo in itertools.product(range(m), repeat=n):
        berth_of = {i: combo[i] for i in range(n)}
        _search_assignment(inst, berth_of, order, objective, best, deadline,
                           max_cases)
        if best.truncated:
            break

    wall = time.perf_counter() - t0
    if best.truncated:
        return ExhaustiveResult(
            None if best.assign is None else best.obj, best.assign, best.cases,
            wall, feasible=False,
            note=f"hit the {time_budget}s time budget or the {max_cases:.1e} case "
                 f"budget after {best.cases} expansions; may not be optimal")
    if best.assign is None:
        return ExhaustiveResult(None, None, best.cases, wall, feasible=False,
                                note="no feasible schedule found")
    return ExhaustiveResult(best.obj, best.assign, best.cases, wall)


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
