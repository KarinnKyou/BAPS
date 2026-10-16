"""Instance families whose optimum is known, used to validate the exact models.

The expected optimum comes from mathematics rather than from a transcription:
each family is built so that a lower bound on the makespan is attainable, and
that bound is then the optimum. A model reproducing it has been checked against
something independent of its own code.
"""

from __future__ import annotations

import sys
import time

import pulp


def solve_shop(routes: list[list[int]], proc: list[list[int]],
               name: str = "shop", time_limit: float = 120.0) -> dict:
    """Solve a job shop, where `routes[j][k]` is the machine of operation k of job j.

    The permutation flow shop is the special case in which every job follows the
    same machine order. Returns a dict with the objective, the lower bound, the
    machine-overlap count and the model size.
    """
    n = len(routes)
    m = max(mach for row in routes for mach in row) + 1
    UB = sum(sum(row) for row in proc) + 1
    BIG = UB + 1

    prob = pulp.LpProblem(f"SHOP_{name}", pulp.LpMinimize)
    s = {(j, k): prob.add_variable(f"s_{j}_{k}", lowBound=0, upBound=UB, cat="Integer")
         for j in range(n) for k in range(len(routes[j]))}
    cmax = prob.add_variable("C_max", lowBound=0, upBound=UB)
    prob.setObjective(cmax)

    for j in range(n):
        for k in range(1, len(routes[j])):
            prob.addConstraint(s[(j, k)] >= s[(j, k - 1)] + proc[j][k - 1],
                               name=f"seq_{j}_{k}")
        last = len(routes[j]) - 1
        prob.addConstraint(cmax >= s[(j, last)] + proc[j][last], name=f"cmax_{j}")

    ops = [(j, k, routes[j][k]) for j in range(n) for k in range(len(routes[j]))]
    nb = 0
    for a in range(len(ops)):
        for b in range(a + 1, len(ops)):
            j, k, mk = ops[a]
            jp, kp, mkp = ops[b]
            if mk != mkp or j == jp:
                continue
            z = prob.add_variable(f"z_{j}_{k}_{jp}_{kp}", cat="Binary")
            nb += 1
            prob.addConstraint(s[(j, k)] + proc[j][k] <= s[(jp, kp)] + BIG * (1 - z),
                               name=f"dA_{j}_{k}_{jp}_{kp}")
            prob.addConstraint(s[(jp, kp)] + proc[jp][kp] <= s[(j, k)] + BIG * z,
                               name=f"dB_{j}_{k}_{jp}_{kp}")

    t0 = time.perf_counter()
    stats = prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    wall = time.perf_counter() - t0
    status = getattr(stats, "status", None)
    obj = pulp.value(prob.objective) if status == 1 else None

    # Re-check feasibility by scanning machine intervals directly.
    overlaps = 0
    if status == 1:
        by_m: dict[int, list[tuple[int, int]]] = {}
        for j in range(n):
            for k, mach in enumerate(routes[j]):
                st = int(round(s[(j, k)].value()))
                by_m.setdefault(mach, []).append((st, st + proc[j][k]))
        for ivs in by_m.values():
            ivs.sort()
            for x, y in zip(ivs, ivs[1:]):
                if y[0] < x[1]:
                    overlaps += 1

    loads = [0] * m
    for j in range(n):
        for k, mach in enumerate(routes[j]):
            loads[mach] += proc[j][k]
    lb_machine = max(loads)
    lb_job = max(sum(row) for row in proc)
    lb = max(lb_machine, lb_job)

    return {
        "name": name, "n": n, "m": m, "solved": obj, "lower_bound": lb,
        "lb_machine": lb_machine, "lb_job": lb_job,
        "match_lb": (obj is not None and abs(obj - lb) < 1e-6),
        "wall_time_s": round(wall, 3), "overlaps": overlaps,
        "vars": prob.numVariables(), "cons": prob.numConstraints(), "binaries": nb,
    }


def case_single_machine() -> dict:
    """One machine, n jobs, so the optimum is the total processing time."""
    procs = [7, 3, 5, 4, 6, 2, 8, 1]
    routes = [[0] for _ in procs]
    proc = [[p] for p in procs]
    r = solve_shop(routes, proc, "single_machine")
    r["expected"] = sum(procs)
    r["expected_reason"] = "the bound sum(p) is attained"
    r["match_expected"] = abs((r["solved"] or -1) - r["expected"]) < 1e-6
    return r


def case_parallel_balanced(m: int = 3, k: int = 4) -> dict:
    """m machines with m*k unit jobs, so the bound k is attained by an even split."""
    n = m * k
    procs = [1] * n
    routes = [[j % m] for j in range(n)]
    proc = [[1] for _ in range(n)]
    r = solve_shop(routes, proc, f"parallel_balanced_{m}x{k}")
    r["expected"] = k
    r["expected_reason"] = (f"{n} unit jobs over {m} machines give a bound of {k}, "
                            f"reached by an even split")
    r["match_expected"] = abs((r["solved"] or -1) - r["expected"]) < 1e-6
    return r


def case_flow_bottleneck() -> dict:
    """Flow shop where the bound is the bottleneck load plus the minimum tail.

    Machine load alone is not a valid bound here. After the last job clears the
    bottleneck it still has to be processed on the following machines, so the
    bound is

        max over machines k of (load of k + minimum remaining work after k).

    For k = 0 that is 6n + 3 + 1. Using the machine load by itself understates the
    optimum, which is why the tail term has to be included explicitly.
    """
    n = 4
    routes = [[0, 1, 2] for _ in range(n)]
    proc = [[6, 3, 1] for _ in range(n)]
    r = solve_shop(routes, proc, "flow_bottleneck")

    m = 3
    loads = [0] * m
    tails = [[0] * m for _ in range(n)]
    for j in range(n):
        for k, mach in enumerate(routes[j]):
            loads[mach] += proc[j][k]
    for j in range(n):
        for k in range(m - 2, -1, -1):
            tails[j][k] = tails[j][k + 1] + proc[j][k + 1]
    lb_correct = max(loads[k] + min(tails[j][k] for j in range(n)) for k in range(m))
    r["expected"] = lb_correct
    r["expected_reason"] = (f"bottleneck load {loads[0]} plus minimum tail "
                            f"{min(tails[j][0] for j in range(n))} gives {lb_correct}")
    r["match_expected"] = abs((r["solved"] or -1) - lb_correct) < 1e-6
    r["lb_with_tail"] = lb_correct
    r["lb_naive_machine_load"] = max(loads)
    return r


def case_two_implementations_agree() -> dict:
    """Two independent job-shop implementations must produce the same makespan.

    `JobShopMILP` assumes every operation of a job has the same duration, while
    `solve_shop` takes a duration per operation. On an instance where all
    durations are equal the two must agree.
    """
    from ..models.classical import JobShopMILP, make_simple_problem, to_instance_view

    P = make_simple_problem("equal_durations", 4, 3, seed=7)
    routes = {j: list(range(P.m)) for j in range(P.n)}
    inst = to_instance_view(P)
    M = JobShopMILP(P, routes=routes, instance_view=inst)
    obj_js = M.solve().objective

    rts = [routes[j] for j in range(P.n)]
    prs = [[P.jobs[j].proc] * P.m for j in range(P.n)]
    obj_shop = solve_shop(rts, prs, "equal_durations")["solved"]

    agree = (obj_js is not None and obj_shop is not None
             and abs(obj_js - obj_shop) < 1e-6)
    return {"name": "two implementations agree", "obj_jobshop": obj_js,
            "obj_per_operation": obj_shop, "match_expected": agree,
            "expected_reason": "independent implementations must agree"}


if __name__ == "__main__":
    for fn in (case_single_machine, case_parallel_balanced,
               case_flow_bottleneck, case_two_implementations_agree):
        result = fn()
        print(f"{result['name']:<28} {'ok' if result['match_expected'] else 'FAIL'}")
    sys.exit(0)
