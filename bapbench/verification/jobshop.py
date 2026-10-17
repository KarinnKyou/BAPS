"""Job-shop optimum by branch and bound over active schedules, no solver.

Independent of the MILP formulation, so agreement with it is evidence rather than
self-consistency.

The branching rule is Giffler and Thompson's (1960): at each node, take the
unscheduled operation that can finish earliest, and branch on which operation of
that machine's *conflict set* -- the operations on that machine that could start
before the chosen one finishes -- is scheduled next. Every active schedule is a
path in that tree, and an optimal schedule is always active, so exploring the whole
tree is exact.

This replaces a search that branched only on operations whose earliest start
equalled the global minimum, which generates the *non-delay* schedules. That set is
a proper subset of the active ones, so the old search was only an upper bound on
the makespan: on 60 random job shops with per-operation durations and shuffled
routes it reported a larger value on 6 (e.g. 27 against an optimum of 24).

The tree is pruned by a lower bound on the makespan of any completion: the
remaining work of each job after its machine becomes free, and the remaining work
on each machine after that machine becomes free. That bound is weak -- it ignores
queueing on a machine -- so the practical envelope is small: up to about six jobs
on three machines is solved in milliseconds, eight on three needs roughly half a
minute, and eight on four does not finish within a minute. The instance families
this package builds (equal durations, identity routes) are the worst case, because
many distinct schedules have the same makespan. Larger instances hit the time
budget and report that instead of returning a value.
"""

from __future__ import annotations

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


def _remaining_work(proc: list[list[int]]) -> list[list[int]]:
    """``remaining[j][k]``: work left in job ``j`` from operation ``k`` onward.

    A job's operations are sequential, so all of it must follow the job's current
    completion time; this is the job half of the lower bound.
    """
    out = []
    for row in proc:
        tail = [0] * (len(row) + 1)
        for k in range(len(row) - 1, -1, -1):
            tail[k] = tail[k + 1] + row[k]
        out.append(tail)
    return out


def job_shop_optimum(routes: list[list[int]], proc: list[list[int]],
                     time_budget: float = 120.0) -> JobShopOptimum:
    """Exact makespan of a job shop, or a report of why it could not be proved.

    ``routes[j][k]`` is the machine of operation ``k`` of job ``j``, ``proc[j][k]``
    its duration. A route may be shorter than the machine count and may revisit a
    machine.

    ``feasible=True`` with an empty ``note`` means the search finished, so the
    value is the optimum. A run that hits the time budget returns its best
    incumbent with ``feasible=False`` and a note: that value is an upper bound and
    must not be compared as if it were the optimum.
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget
    n = len(routes)
    if n == 0:
        return JobShopOptimum(0.0, [], 0, time.perf_counter() - t0)

    for j, row in enumerate(routes):
        if not row:
            raise ValueError(f"job {j} has an empty route")
        if len(row) != len(proc[j]):
            raise ValueError(
                f"job {j} has {len(row)} operations but {len(proc[j])} durations")
        if any(p <= 0 for p in proc[j]):
            raise ValueError(f"job {j} has a non-positive duration; the conflict "
                             f"set of Giffler-Thompson assumes positive durations")

    m = max(mach for row in routes for mach in row) + 1
    remaining = _remaining_work(proc)

    # Work still to run on each machine. Both this and the per-job term are lower
    # bounds on the makespan, so the larger of the two is one as well.
    total_on_machine = [0] * m
    for j in range(n):
        for k, mach in enumerate(routes[j]):
            total_on_machine[mach] += proc[j][k]
    lower_bound = max(max(total_on_machine), max(sum(row) for row in proc))

    machine_free = [0] * m
    job_free = [0] * n
    next_op = [0] * n
    done_on_machine = [0] * m
    seq: list[tuple[int, int]] = []
    best: dict = {"value": None, "seq": None}
    nodes = 0
    timed_out = False

    def bound() -> int:
        """Lower bound on the makespan of any completion of the current node."""
        lb = lower_bound
        for j in range(n):
            k = next_op[j]
            if k < len(routes[j]):
                mach = routes[j][k]
                start = max(job_free[j], machine_free[mach])
                lb = max(lb, start + remaining[j][k])
        for mach in range(m):
            lb = max(lb, machine_free[mach]
                     + total_on_machine[mach] - done_on_machine[mach])
        return lb

    def rec() -> None:
        nonlocal nodes, timed_out
        nodes += 1
        if nodes % 512 == 0 and time.perf_counter() > deadline:
            timed_out = True
            return

        info = []                       # (earliest completion, earliest start, ...)
        for j in range(n):
            k = next_op[j]
            if k >= len(routes[j]):
                continue
            mach = routes[j][k]
            est = max(job_free[j], machine_free[mach])
            info.append((est + proc[j][k], est, -remaining[j][k],
                         j, k, mach))
        if not info:
            makespan = max(job_free)
            if best["value"] is None or makespan < best["value"]:
                best["value"] = makespan
                best["seq"] = list(seq)
            return

        if best["value"] is not None and bound() >= best["value"]:
            return

        # Giffler-Thompson: the operation that can finish earliest fixes the
        # machine to branch on; the conflict set is every operation on that
        # machine that could start before that operation finishes.
        ect_star, _est_star, _neg, j_star, k_star, m_star = min(info)
        conflict = [t for t in info if t[5] == m_star and t[1] < ect_star]
        if not conflict:
            # Only reachable for a zero-duration operation, which is rejected
            # above; keeping the node branching rather than silently pruning it
            # makes the completeness argument independent of that validation.
            conflict = [(ect_star, _est_star, 0, j_star, k_star, m_star)]

        for (_ect, est, _rem, j, k, mach) in sorted(conflict):
            end = est + proc[j][k]
            old_machine, old_job = machine_free[mach], job_free[j]
            machine_free[mach] = end
            job_free[j] = end
            next_op[j] = k + 1
            done_on_machine[mach] += proc[j][k]
            seq.append((j, k))

            rec()

            seq.pop()
            done_on_machine[mach] -= proc[j][k]
            next_op[j] = k
            machine_free[mach] = old_machine
            job_free[j] = old_job
            if timed_out:
                return

    rec()
    wall = time.perf_counter() - t0

    if timed_out:
        return JobShopOptimum(best["value"], best["seq"], nodes, wall, feasible=False,
                              note=f"exceeded the {time_budget}s budget after "
                                   f"{nodes} nodes; the value is an upper bound")
    if best["value"] is None:
        return JobShopOptimum(None, None, nodes, wall, feasible=False,
                              note="no schedule found")
    return JobShopOptimum(float(best["value"]), best["seq"], nodes, wall)


if __name__ == "__main__":
    from ..models.classical import make_simple_problem

    for n, m in ((4, 3), (4, 4), (5, 3), (6, 3)):
        prob = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
        routes = [list(range(m)) for _ in range(n)]
        proc = [[job.proc] * m for job in prob.jobs]
        r = job_shop_optimum(routes, proc, time_budget=30.0)
        print(f"  {n}x{m}: optimum={r.objective} nodes={r.nodes} "
              f"{r.wall_time:.3f}s {r.note}")
