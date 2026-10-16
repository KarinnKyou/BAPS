# -*- coding: utf-8 -*-
"""Metaheuristics for the berth allocation problem.

A solution is a permutation of vessels, decoded by the shared placement
routine in bapbench.heuristics.constructive; all algorithms search that
space under the same evaluation, so their results are comparable.

local_search is first-improvement hill climbing over swap, relocate and
reverse; ILS adds random perturbation and keeps the best solution seen;
ALNS is destroy-and-repair with roulette-wheel adaptive operator weights
over random and worst removal destroy and greedy and random insertion.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from ..instances import Instance, Schedule
from .constructive import place_in_order, evaluate


@dataclass
class Evaluator:
    """Permutation evaluator caching objective values and counting calls."""
    inst: Instance
    objective: str = "sum_C"
    rule: str = "earliest_start"
    _cache: dict = field(default_factory=dict)
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0

    def __call__(self, order: list[int]) -> float:
        key = tuple(order)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.calls += 1
        try:
            _, val = evaluate(self.inst, list(key), self.rule, self.objective)
            val = float(val)
        except RuntimeError:
            # infeasible within the horizon: penalise rather than discard
            self.failures += 1
            val = math.inf
        self._cache[key] = val
        return val

    def stats(self) -> dict:
        return {"evals": self.calls, "cache_hits": self.cache_hits,
                "infeasible_evals": self.failures}


def _finish(inst: Instance, order: list[int], name: str, t0: float,
            objective: str, ev: Evaluator, extra: dict | None = None) -> Schedule:
    a = place_in_order(inst, order, "earliest_start")
    sch = Schedule(assign=a, solver=name, wall_time=time.perf_counter() - t0,
                   extra={**(extra or {}), **ev.stats(), "order": order})
    sch.check(inst)
    sch.compute_objective(inst, objective)
    return sch


def nbr_swap(order: list[int], rng: random.Random) -> list[int]:
    n = len(order)
    if n < 2:
        return order[:]
    i, j = rng.sample(range(n), 2)
    o = order[:]
    o[i], o[j] = o[j], o[i]
    return o


def nbr_relocate(order: list[int], rng: random.Random) -> list[int]:
    n = len(order)
    if n < 2:
        return order[:]
    i = rng.randrange(n)
    o = order[:]
    v = o.pop(i)
    j = rng.randrange(len(o) + 1)
    o.insert(j, v)
    return o


def nbr_reverse(order: list[int], rng: random.Random) -> list[int]:
    n = len(order)
    if n < 3:
        return nbr_swap(order, rng)
    i, j = sorted(rng.sample(range(n), 2))
    if i == j:
        return order[:]
    return order[:i] + order[i:j + 1][::-1] + order[j + 1:]


NEIGHBORHOODS = {"swap": nbr_swap, "relocate": nbr_relocate, "reverse": nbr_reverse}


def local_search(inst: Instance, order: list[int], ev: Evaluator,
                 rng: random.Random, *, max_iters: int = 2000,
                 move: str = "relocate") -> tuple[list[int], float]:
    fn = NEIGHBORHOODS[move]
    cur = order[:]
    cur_val = ev(cur)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        for _ in range(max(1, len(cur))):
            it += 1
            if it >= max_iters:
                break
            cand = fn(cur, rng)
            cv = ev(cand)
            if cv < cur_val - 1e-9:
                cur, cur_val, improved = cand, cv, True
                break
    return cur, cur_val


def _best_seed(inst: Instance, objective: str, ev: Evaluator
               ) -> tuple[list[int], float]:
    """Cheapest decodable starting permutation, from the constructive rules.

    The rules can all fail to decode on a tight instance, in which case every
    candidate evaluates to infinity. Returning None to the search then crashes
    with an unhelpful TypeError, so this falls back to the decoders with a restart
    budget and reports clearly if even those cannot place every vessel.
    """
    from ..heuristics.constructive import (
        PLACE_RESTARTS_BY_METHOD, fcfs, lpt, spt,
    )

    candidates: list[list[int]] = []
    for fn in (lpt, fcfs, spt):
        name = fn.__name__.upper()
        try:
            sch = fn(inst, objective)
        except RuntimeError:
            continue
        order = sch.extra.get("order")
        if order and sch.feasible:
            candidates.append(list(order))

    best, best_val = None, math.inf
    for order in candidates:
        v = ev(order)
        if v < best_val:
            best, best_val = order[:], v

    if best is None:
        # Every rule failed; a plain order plus the restart budget is the last
        # resort, and it raises with the per-vessel windows if that fails too.
        fallback = list(range(inst.n))
        place_in_order(inst, fallback, "earliest_start",
                       restarts=max(PLACE_RESTARTS_BY_METHOD.values()))
        v = ev(fallback)
        if v == math.inf:
            raise RuntimeError(
                f"no decodable starting permutation for {inst.name}: the "
                f"constructive rules and the restart budget all failed to place "
                f"every vessel")
        best, best_val = fallback, v
    return best, best_val


def iterated_local_search(inst: Instance, objective: str = "sum_C",
                          *, iters: int = 60, seed: int = 20261014,
                          kicks: int = 3, moves=("relocate", "swap", "reverse")) -> Schedule:
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ev = Evaluator(inst, objective)
    n = inst.n

    cur = list(range(n))
    best, best_val = _best_seed(inst, objective, ev)

    cur, cur_val = local_search(inst, best, ev, rng, move=random.Random(seed).choice(moves))
    # The declared best must include the local optimum just computed, or a run that
    # improves on its seed and then finds nothing better returns the seed value.
    best, best_val = cur[:], cur_val

    for _ in range(iters):
        cand = cur[:]
        for _ in range(kicks):
            cand = NEIGHBORHOODS[rng.choice(moves)](cand, rng)
        mv = rng.choice(moves)
        cand, cand_val = local_search(inst, cand, ev, rng, move=mv)
        if cand_val < cur_val - 1e-9:
            cur, cur_val = cand, cand_val
            if cand_val < best_val - 1e-9:
                best, best_val = cand[:], cand_val
        elif cand_val < cur_val + 1e-9:
            # accept equal-cost moves to escape plateaus
            cur, cur_val = cand, cand_val

    return _finish(inst, best, "ILS", t0, objective, ev,
                   {"iters": iters, "kicks": kicks})


def _destroy_random(order: list[int], q: int, rng: random.Random) -> tuple[list[int], list[int]]:
    idx = sorted(rng.sample(range(len(order)), min(q, len(order))))
    removed = [order[i] for i in idx]
    rest = [v for i, v in enumerate(order) if i not in set(idx)]
    return rest, removed


def _destroy_worst(order: list[int], q: int, rng: random.Random,
                   inst: Instance, ev: Evaluator) -> tuple[list[int], list[int]]:
    scored = sorted(order, key=lambda j: -inst.vessels[j].handle)
    removed = scored[:min(q, len(order))]
    rest = [v for v in order if v not in set(removed)]
    return rest, removed


def _repair_greedy(rest: list[int], removed: list[int], ev: Evaluator,
                   rng: random.Random) -> list[int]:
    cur = rest[:]
    for v in removed:
        best_pos, best_val = 0, math.inf
        for pos in range(len(cur) + 1):
            cand = cur[:pos] + [v] + cur[pos:]
            val = ev(cand)
            if val < best_val - 1e-9:
                best_pos, best_val = pos, val
        cur.insert(best_pos, v)
    return cur


def _repair_random(rest: list[int], removed: list[int], ev: Evaluator,
                   rng: random.Random) -> list[int]:
    cur = rest[:]
    rng.shuffle(removed)
    for v in removed:
        pos = rng.randrange(len(cur) + 1)
        cur.insert(pos, v)
    return cur


DESTROY = {"random": _destroy_random, "worst": _destroy_worst}
REPAIR = {"greedy": _repair_greedy, "random": _repair_random}


def alns(inst: Instance, objective: str = "sum_C", *, iters: int = 120,
         seed: int = 20261014, q_frac: float = 0.3,
         t0_temp: float = 5.0, cooling: float = 0.995) -> Schedule:
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ev = Evaluator(inst, objective)
    n = inst.n
    q = max(1, int(round(q_frac * n)))

    from ..heuristics.constructive import fcfs, lpt
    cur = fcfs(inst, objective).extra.get("order", list(range(n)))[:]
    cur_val = ev(cur)
    best, best_val = cur[:], cur_val

    dw = {k: 1.0 for k in DESTROY}
    rw = {k: 1.0 for k in REPAIR}
    score = {("d", k): 0.0 for k in DESTROY}
    score.update({("r", k): 0.0 for k in REPAIR})
    used = {("d", k): 0 for k in DESTROY}
    used.update({("r", k): 0 for k in REPAIR})

    def roulette(w: dict, rng: random.Random) -> str:
        tot = sum(w.values())
        r, acc = rng.random() * tot, 0.0
        for k, v in w.items():
            acc += v
            if r <= acc:
                return k
        return next(iter(w))

    T = t0_temp
    for _ in range(iters):
        dk = roulette(dw, rng)
        rk = roulette(rw, rng)
        used[("d", dk)] += 1
        used[("r", rk)] += 1

        if dk == "worst":
            rest, removed = DESTROY[dk](cur, q, rng, inst, ev)
        else:
            rest, removed = DESTROY[dk](cur, q, rng)
        cand = REPAIR[rk](rest, removed, ev, rng)
        cand_val = ev(cand)

        gain = 0.0
        accepted = False
        if cand_val < best_val - 1e-9:
            best, best_val = cand[:], cand_val
            gain = 3.0
            accepted = True
        elif cand_val < cur_val - 1e-9:
            gain = 2.0
            accepted = True
        elif rng.random() < math.exp(-(cand_val - cur_val) / max(T, 1e-9)):
            gain = 1.0
            accepted = True
        if accepted:
            cur, cur_val = cand, cand_val
        score[("d", dk)] += gain
        score[("r", rk)] += gain
        T *= cooling

        if _ % 20 == 19:
            for key in dw:
                u = used[("d", key)]
                if u:
                    dw[key] = max(0.05, 0.7 * dw[key] + 0.3 * (score[("d", key)] / u))
            for key in rw:
                u = used[("r", key)]
                if u:
                    rw[key] = max(0.05, 0.7 * rw[key] + 0.3 * (score[("r", key)] / u))

    return _finish(inst, best, "ALNS", t0, objective, ev,
                   {"iters": iters, "q": q, "destroy_w": dw, "repair_w": rw})


HEURISTICS = {
    "LS": lambda inst, obj, seed: _ls_entry(inst, obj, seed),
    "ILS": lambda inst, obj, seed: iterated_local_search(inst, obj, seed=seed),
    "ALNS": lambda inst, obj, seed: alns(inst, obj, seed=seed),
}


def _ls_entry(inst: Instance, objective: str, seed: int) -> Schedule:
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ev = Evaluator(inst, objective)
    best, best_val = _best_seed(inst, objective, ev)
    best, best_val = local_search(inst, best, ev, rng, move="relocate")
    return _finish(inst, best, "LS", t0, objective, ev)


def run_heuristics(inst: Instance, objective: str = "sum_C",
                   seed: int = 20261014) -> dict[str, Schedule]:
    out = {}
    for name, fn in HEURISTICS.items():
        try:
            out[name] = fn(inst, objective, seed)
        except Exception as e:                                   # noqa: BLE001
            out[name] = Schedule(assign={}, feasible=False, objective=None,
                                 violations=[f"{type(e).__name__}: {e}"], solver=name)
    return out
