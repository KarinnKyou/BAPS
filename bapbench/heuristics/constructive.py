# -*- coding: utf-8 -*-
"""Constructive baselines: FCFS, LPT, SPT, greedy, simulated annealing, GA.

Every method searches over permutations of vessels and shares one placement
routine, so differences between methods come only from the priority rule or the
search, not from the decoding.
"""

from __future__ import annotations

import math
import random
import time

from ..instances import Instance, Schedule



def _earliest_feasible_start(inst: Instance, i: int, k: int, not_before: int) -> int | None:
    """Earliest start >= not_before for vessel i at berth k that avoids closures.

    Returns None when the horizon holds no feasible start, or when the vessel is
    longer than the berth: a vessel occupies the whole berth, so it can only berth
    where its length fits. The exact models, the enumerator and both feasibility
    checkers enforce that, so the decoder has to as well or it would produce a
    schedule the package's own check rejects.
    """
    v = inst.vessels[i]
    if v.length > inst.berths[k].length:
        return None
    latest = inst.horizon - v.handle
    t = max(v.arrive, not_before)
    while t <= latest:
        if all(not inst.blocked(k, tt) for tt in range(t, t + v.handle)):
            return t
        t += 1
    return None


def _place_once(inst: Instance, order: list[int], rule: str) -> dict[int, tuple[int, int]] | None:
    """One greedy placement pass; returns None if a vessel cannot be placed."""
    n, m = inst.n, inst.m
    free = [0] * m
    crane_busy = [0] * inst.horizon
    assign: dict[int, tuple[int, int]] = {}

    def crane_ok(st: int, p: int) -> bool:
        """Whether crane capacity suffices on every period of [st, st+p).

        Uses inst.crane_capacity(t) rather than the quay_crane_limit constant,
        because crane failures make capacity vary by period.
        """
        for t in range(st, min(st + p, inst.horizon)):
            cap = inst.crane_capacity(t)
            if cap is not None and crane_busy[t] + 1 > cap:
                return False
        return True

    for i in order:
        v = inst.vessels[i]
        best = None
        for k in range(m):
            st = _earliest_feasible_start(inst, i, k, free[k])
            while st is not None and not crane_ok(st, v.handle):
                st = _earliest_feasible_start(inst, i, k, st + 1)
            if st is None:
                continue
            if rule == "earliest_start":
                key = (st, st + v.handle, k)
            elif rule == "earliest_finish":
                key = (st + v.handle, st, k)
            elif rule == "first_fit":
                key = (k, st)
            elif rule == "least_crane_load":
                # Prefer the slot whose periods are least contended. Always
                # taking the earliest start packs vessels into the early periods
                # and leaves later ones with no crane-legal slot.
                load = sum(crane_busy[t]
                           for t in range(st, min(st + v.handle, inst.horizon)))
                key = (load, st, k)
            else:
                raise ValueError(f"unknown placement rule: {rule}")
            if best is None or key < best[0]:
                best = (key, k, st)
        if best is None:
            return None                       # this pass stranded vessel i
        _, k, st = best
        assign[i] = (k, st)
        free[k] = st + v.handle
        for t in range(st, min(st + v.handle, inst.horizon)):
            crane_busy[t] += 1

    return assign


# Retrying the same permutation under different placement orientations.
# The quay-crane limit couples all vessels globally, so a single greedy pass can
# strand a vessel even when a feasible schedule exists: packing earlier vessels
# into early periods may leave no crane-legal slot, while another orientation
# (later start, other berth) still fits. An orientation changes only how a given
# permutation is decoded, not the permutation itself.
_PLACE_ATTEMPTS = ("earliest_start", "earliest_finish", "first_fit",
                   "least_crane_load")

# Shuffled permutations to try when the given order cannot be decoded. Only about
# 1-5% of permutations decode on a tight instance, so a failed first attempt says
# little about whether a schedule exists. The draws are seeded, so a heuristic
# stays deterministic. Budgets differ because success rates do: on a tight
# 20-vessel instance, longest-handling-first needs about 50 restart orders while
# arrival order needs about 1000, so a uniform budget would either cost the first
# heuristic 2.7 s or fail the second.
PLACE_RESTARTS = 100
PLACE_RESTARTS_BY_METHOD = {
    "FCFS": 600,
    "LPT": 100,
    "SPT": 250,
    "GREEDY": 300,
}


def _place_deferring(inst: Instance, order: list[int], rule: str) -> dict[int, tuple[int, int]]:
    """Greedy pass that carries stranded vessels and retries them at the end.

    A single forward pass can strand a vessel because the crane limit couples all
    vessels globally: an early vessel may take the only crane-legal slot a later
    one needed. Placing the rest first and retrying the stranded ones afterwards
    recovers many such cases without changing the permutation.
    """
    assign: dict[int, tuple[int, int]] = {}
    remaining = list(order)
    for _round in range(3):
        if not remaining:
            break
        got = _place_once_partial(inst, remaining, rule, assign)
        if not got:
            break
        remaining = got
    return assign if len(assign) == inst.n else {}


def _place_once_partial(inst: Instance, order: list[int], rule: str,
                        assign: dict[int, tuple[int, int]]) -> list[int]:
    """Place what can be placed; return the vessels that could not be placed."""
    m = inst.m
    free = [0] * m
    for (_vid, (k, st)) in assign.items():
        free[k] = max(free[k], st + inst.vessels[_vid].handle)
    crane_busy = [0] * inst.horizon
    for vid, (_k, st) in assign.items():
        for t in range(st, min(st + inst.vessels[vid].handle, inst.horizon)):
            crane_busy[t] += 1

    def crane_ok(st: int, p: int) -> bool:
        for t in range(st, min(st + p, inst.horizon)):
            cap = inst.crane_capacity(t)
            if cap is not None and crane_busy[t] + 1 > cap:
                return False
        return True

    left: list[int] = []
    for i in order:
        v = inst.vessels[i]
        best = None
        for k in range(m):
            st = _earliest_feasible_start(inst, i, k, free[k])
            while st is not None and not crane_ok(st, v.handle):
                st = _earliest_feasible_start(inst, i, k, st + 1)
            if st is None:
                continue
            if rule == "earliest_finish":
                key = (st + v.handle, st, k)
            elif rule == "first_fit":
                key = (k, st)
            else:
                key = (st, st + v.handle, k)
            if best is None or key < best[0]:
                best = (key, k, st)
        if best is None:
            left.append(i)
            continue
        _, k, st = best
        assign[i] = (k, st)
        free[k] = st + v.handle
        for t in range(st, min(st + v.handle, inst.horizon)):
            crane_busy[t] += 1
    return left


def _room_of(inst: Instance, i: int) -> int:
    """Latest feasible start minus arrival: how much room vessel i has."""
    v = inst.vessels[i]
    latest = 0
    for k in range(inst.m):
        for t in range(v.arrive, inst.horizon - v.handle + 1):
            if all(not inst.blocked(k, tt) for tt in range(t, t + v.handle)):
                latest = max(latest, t)
    return latest - v.arrive


def place_in_order(inst: Instance, order: list[int],
                   rule: str = "earliest_start",
                   restarts: int = 0) -> dict[int, tuple[int, int]]:
    """Place vessels in the given order; returns assign[v] = (berth_id, start_period).

    Enforces both operational constraints:
      * tidal windows: a handling interval may not straddle a closure
      * quay cranes  : concurrent vessels in any period <= quay_crane_limit

    `rule` is the preferred orientation; on failure the remaining entries of
    _PLACE_ATTEMPTS are retried in turn, then each orientation is retried with
    stranded vessels deferred to the end.

    Between 1% and 5% of permutations decode successfully on a tight instance,
    so a single permutation can fail even when a schedule exists. `restarts`
    additionally tries that many shuffled permutations, drawn from a generator
    seeded by the instance name and the order, so a given call remains
    reproducible.

    RuntimeError is raised only when every attempt fails, and its message gives
    the per-vessel feasible windows so a genuinely infeasible instance is
    distinguishable from a heuristic failure.
    """
    attempts = [rule] + [r for r in _PLACE_ATTEMPTS if r != rule]
    for r in attempts:
        got = _place_once(inst, order, r)
        if got is not None and len(got) == inst.n:
            return got
    for r in attempts:
        got = _place_deferring(inst, order, r)
        if len(got) == inst.n:
            return got

    if restarts > 0:
        rng = random.Random(f"{inst.name}|{tuple(order)}|{rule}")
        for _ in range(restarts):
            shuffled = list(order)
            rng.shuffle(shuffled)
            for r in attempts:
                got = _place_once(inst, shuffled, r)
                if got is not None and len(got) == inst.n:
                    return got
            for r in attempts:
                got = _place_deferring(inst, shuffled, r)
                if len(got) == inst.n:
                    return got

    # Every orientation failed: give windows so a heuristic failure is not
    # mistaken for an infeasible instance.
    diag = []
    Q = inst.quay_crane_limit
    for v in inst.vessels:
        wins = []
        for k in range(inst.m):
            st = _earliest_feasible_start(inst, v.vid, k, v.arrive)
            wins.append("none" if st is None else str(st))
        diag.append(f"{v.name}(arrive {v.arrive}, handle {v.handle}): "
                    f"earliest feasible by berth={wins}")
    raise RuntimeError(
        "constructive placement failed (no orientation of {0} placed all {1} vessels)"
        "; the instance may still be feasible: quay-crane limit {2} can strand "
        "the greedy passes. Per-vessel diagnostics: {3}".format(
            len(attempts), inst.n, Q, " | ".join(diag))
    )


def evaluate(inst: Instance, order: list[int], rule: str = "earliest_start",
             objective: str = "sum_C") -> tuple[dict[int, tuple[int, int]], float]:
    a = place_in_order(inst, order, rule)
    sch = Schedule(assign=a)
    sch.compute_objective(inst, objective)
    return a, sch.objective



# Baselines

def _finish(inst, assign, name, t0, objective, extra=None) -> Schedule:
    sch = Schedule(assign=assign, solver=name, wall_time=time.perf_counter() - t0,
                   extra=extra or {})
    sch.check(inst)
    sch.compute_objective(inst, objective)
    return sch


def fcfs(inst: Instance, objective: str = "sum_C") -> Schedule:
    """First come, first served: ascending arrival time."""
    t0 = time.perf_counter()
    order = sorted(range(inst.n), key=lambda i: (inst.vessels[i].arrive, inst.vessels[i].vid))
    a = place_in_order(inst, order, "earliest_start",
                       restarts=PLACE_RESTARTS_BY_METHOD["FCFS"])
    return _finish(inst, a, "FCFS", t0, objective, {"order": order})


def lpt(inst: Instance, objective: str = "sum_C") -> Schedule:
    """Longest handling time first (classic rule for parallel machines)."""
    t0 = time.perf_counter()
    order = sorted(range(inst.n), key=lambda i: (-inst.vessels[i].handle, inst.vessels[i].vid))
    a = place_in_order(inst, order, "first_fit",
                       restarts=PLACE_RESTARTS_BY_METHOD["LPT"])
    return _finish(inst, a, "LPT", t0, objective, {"order": order})


def spt(inst: Instance, objective: str = "sum_C") -> Schedule:
    """Shortest handling time first."""
    t0 = time.perf_counter()
    order = sorted(range(inst.n), key=lambda i: (inst.vessels[i].handle, inst.vessels[i].vid))
    a = place_in_order(inst, order, "earliest_start",
                       restarts=PLACE_RESTARTS_BY_METHOD["SPT"])
    return _finish(inst, a, "SPT", t0, objective, {"order": order})


def greedy_earliest(inst: Instance, objective: str = "sum_C") -> Schedule:
    """Greedy earliest completion, applied to an arrival-ordered list."""
    t0 = time.perf_counter()
    order = sorted(range(inst.n), key=lambda i: (inst.vessels[i].arrive, inst.vessels[i].vid))
    a = place_in_order(inst, order, "earliest_finish",
                       restarts=PLACE_RESTARTS_BY_METHOD["GREEDY"])
    return _finish(inst, a, "GREEDY", t0, objective, {"order": order})


def _sa_like(inst: Instance, objective: str, *, kind: str,
             iters: int, seed: int, t0_temp: float = 4.0,
             t_end: float = 0.01, pop: int = 30) -> Schedule:
    """Search framework shared by SA and GA; the space is vessel permutations."""
    rng = random.Random(seed)
    t_start = time.perf_counter()
    n = inst.n

    def cost(order):
        try:
            _, c = evaluate(inst, list(order), "earliest_start", objective)
            return c
        except RuntimeError:
            return math.inf

    best_order = list(range(n))
    best_cost = cost(best_order)

    if kind == "SA":
        cur = best_order[:]
        cur_cost = best_cost
        alpha = (t_end / t0_temp) ** (1.0 / max(1, iters))
        T = t0_temp
        for _ in range(iters):
            j, k = rng.sample(range(n), 2) if n >= 2 else (0, 0)
            cand = cur[:]
            cand[j], cand[k] = cand[k], cand[j]
            c = cost(cand)
            if c < cur_cost or rng.random() < math.exp(-(c - cur_cost) / max(T, 1e-9)):
                cur, cur_cost = cand, c
                if c < best_cost:
                    best_order, best_cost = cand[:], c
            T *= alpha

    elif kind == "GA":
        P = [rng.sample(range(n), n) for _ in range(pop)]
        P[0] = best_order[:]
        for _ in range(max(1, iters // pop)):
            scored = sorted(P, key=cost)
            if cost(scored[0]) < best_cost:
                best_order, best_cost = scored[0][:], cost(scored[0])
            # tournament selection, order crossover, swap mutation
            newP = scored[:max(2, pop // 5)]
            while len(newP) < pop:
                a = min(rng.sample(scored[: max(2, pop // 2)], 1), key=cost)
                b = min(rng.sample(scored[: max(2, pop // 2)], 1), key=cost)
                cut = rng.randint(1, n - 1) if n > 1 else 0
                child = a[:cut] + [g for g in b if g not in a[:cut]]
                if n >= 2 and rng.random() < 0.3:
                    j, k = rng.sample(range(n), 2)
                    child[j], child[k] = child[k], child[j]
                newP.append(child)
            P = newP
        if cost(best_order) > min(cost(p) for p in P):
            p = min(P, key=cost)
            best_order, best_cost = p[:], cost(p)
    else:
        raise ValueError(kind)

    a = place_in_order(inst, best_order, "earliest_start")
    return _finish(inst, a, kind, t_start, objective,
                   {"order": best_order, "iters": iters, "seed": seed})


def simulated_annealing(inst: Instance, objective: str = "sum_C",
                        iters: int = 4000, seed: int = 20261014) -> Schedule:
    return _sa_like(inst, objective, kind="SA", iters=iters, seed=seed)


def genetic_algorithm(inst: Instance, objective: str = "sum_C",
                      iters: int = 3000, seed: int = 20261014) -> Schedule:
    return _sa_like(inst, objective, kind="GA", iters=iters, seed=seed)



BASELINES = {
    "FCFS": fcfs,
    "LPT": lpt,
    "SPT": spt,
    "GREEDY": greedy_earliest,
    "SA": simulated_annealing,
    "GA": genetic_algorithm,
}


def run_all_baselines(inst: Instance, objective: str = "sum_C",
                      seed: int = 20261014) -> dict[str, Schedule]:
    out: dict[str, Schedule] = {}
    for name, fn in BASELINES.items():
        try:
            if name in ("SA", "GA"):
                out[name] = fn(inst, objective, seed=seed)
            else:
                out[name] = fn(inst, objective)
        except RuntimeError as e:
            sch = Schedule(assign={}, feasible=False, violations=[str(e)],
                           solver=name, objective=None)
            out[name] = sch
    return out
