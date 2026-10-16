"""Exact job-shop optimum by branch and bound over active schedules, no solver.

Independent of the MILP formulations, so agreement with them is evidence rather
than self-consistency. Searches the set of active schedules, which is guaranteed
to contain an optimum: at each step only operations whose earliest start equals
the current scheduling time are branched on.

Intended for small instances. Larger ones hit the time budget and report that
instead of returning a value.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

@dataclass
class JobShopOptimum:
    objective: float | None
    order: list[tuple[int, int]] | None
    nodes: int
    wall_time: float
    feasible: bool = True
    note: str = ""


def _tails(proc: list[list[int]]) -> list[list[int]]:
    """Remaining work after each operation, within its own job."""
    out = []
    for row in proc:
        m = len(row)
        tail = [0] * m
        for k in range(m - 2, -1, -1):
            tail[k] = tail[k + 1] + row[k + 1]
        out.append(tail)
    return out


def job_shop_optimum(routes: list[list[int]], proc: list[list[int]],
                     time_budget: float = 120.0) -> JobShopOptimum:
    """Minimise C_max exactly, or report why it could not be done.

    routes[j][k] is the machine of operation k of job j; proc[j][k] its duration.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget
    n = len(routes)
    m = max(mach for row in routes for mach in row) + 1
    tails = _tails(proc)

    # machine load is a valid lower bound; so is the longest job
    load = [0] * m
    for j in range(n):
        for k, mach in enumerate(routes[j]):
            load[mach] += proc[j][k]
    lower_bound = max(max(load), max(sum(row) for row in proc))

    machine_free = [0] * m
    job_free = [0] * n
    next_op = [0] * n
    seq: list[tuple[int, int]] = []
    best = {"value": None, "seq": None}
    nodes = 0
    timed_out = False

    def bound() -> int:
        """Earliest finishing time reachable from the current partial schedule."""
        est = 0
        for j in range(n):
            k = next_op[j]
            if k < len(routes[j]):
                mach = routes[j][k]
                start = max(job_free[j], machine_free[mach])
                est = max(est, start + proc[j][k] + tails[j][k])
        return max(est, lower_bound)

    def rec() -> None:
        nonlocal nodes, timed_out
        nodes += 1
        if nodes % 2048 == 0 and time.perf_counter() > deadline:
            timed_out = True
            return

        # candidate operations: those ready to start at the earliest possible time
        ready = []
        earliest = None
        for j in range(n):
            k = next_op[j]
            if k >= len(routes[j]):
                continue
            mach = routes[j][k]
            start = max(job_free[j], machine_free[mach])
            if earliest is None or start < earliest:
                earliest = start
                ready = [(j, k, mach, start)]
            elif start == earliest:
                ready.append((j, k, mach, start))

        if not ready:
            makespan = max(job_free)
            if best["value"] is None or makespan < best["value"]:
                best["value"] = makespan
                best["seq"] = list(seq)
            return

        if best["value"] is not None and bound() >= best["value"]:
            return

        # earliest start first; ties broken by longer remaining work
        ready.sort(key=lambda t: (t[3], -tails[t[0]][t[1]]))
        for (j, k, mach, start) in ready:
            finish = start + proc[j][k]
            old_mf, old_jf = machine_free[mach], job_free[j]
            machine_free[mach] = finish
            job_free[j] = finish
            next_op[j] = k + 1
            seq.append((j, k))

            rec()

            seq.pop()
            next_op[j] = k
            machine_free[mach] = old_mf
            job_free[j] = old_jf
            if timed_out:
                return

    rec()
    wall = time.perf_counter() - t0

    if timed_out:
        return JobShopOptimum(best["value"], best["seq"], nodes, wall, feasible=False,
                              note=f"exceeded the {time_budget}s budget after "
                                   f"{nodes} nodes; value may not be optimal")
    if best["value"] is None:
        return JobShopOptimum(None, None, nodes, wall, feasible=False,
                              note="no schedule found")
    return JobShopOptimum(float(best["value"]), best["seq"], nodes, wall)


if __name__ == "__main__":
    from ..models.classical import make_simple_problem

    for n, m in ((4, 3), (4, 4), (5, 3)):
        prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
        routes = [list(range(m)) for _ in range(n)]
        proc = [[job.proc] * m for job in prob.jobs]
        r = job_shop_optimum(routes, proc, time_budget=30.0)
        print(f"  {n}x{m}: optimum={r.objective} nodes={r.nodes} "
              f"{r.wall_time:.3f}s {r.note}")
    sys.exit(0)
