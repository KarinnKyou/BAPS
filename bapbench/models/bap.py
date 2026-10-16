# -*- coding: utf-8 -*-
"""Exact MILP formulations for the dynamic berth allocation problem.

Imai, Nishimura & Papadimitriou (2001), Transportation Research Part B 35(4).
`sum_i C_i` is the default objective; `C_max` is also supported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pulp

from ..instances import Instance, Schedule

# Solver statuses that mean "a solution may exist but optimality was not proved".
# Treating these as infeasible discards a usable schedule and misleads the caller.
LIMIT_STATUSES = frozenset({
    -4,   # TimeLimit
    -5,   # MemoryLimit
    -6,   # NodeLimit
    -7,   # GapLimit
    -8,   # IterationLimit
    -9,   # SolutionLimit
    -10,  # Interrupted
    -11,  # Stopped
})


def safe_big_m(inst: Instance) -> int:
    """Big-M for the pairwise ordering constraints.

    Too small a value does not make the solver report infeasibility; it returns
    a plausible wrong answer. Computed per instance as `horizon + 1`.
    """
    return inst.horizon + 1


@dataclass
class BAPTimeIndexed:
    """Time-indexed formulation of min sum C_i: O(n*m*T) binaries.

    A vessel occupies its whole berth while handling, so quay capacity reduces
    to "at most one vessel per berth-period" and no disjunctive non-overlap
    rows are needed. Tidal windows and locks restrict the start domains rather
    than adding rows. `solver_kwargs` is forwarded to the solver as a dict, so
    backend option names stay out of the model class.
    """
    inst: Instance
    objective: str = "sum_C"
    full_berth: bool = True     # True: whole berth; relaxing needs interval rows
    time_limit: float | None = None
    mip_gap: float | None = None
    msg: bool = False
    solver_kwargs: dict = field(default_factory=dict)
    # {vid: (berth_id, start_lower_bound, start_fixed)}. A port can commit to a
    # berth, but a start can only slip later, never earlier: start_fixed=False
    # postpones the start, start_fixed=True pins berth and start.
    lock: dict = field(default_factory=dict)
    lock_start_free: bool = False     # read every lock as berth-only

    problem: object = None
    x: dict = None
    U: dict = None
    big_m: int = 0

    def _starts(self, i: int) -> list[int]:
        v = self.inst.vessels[i]
        return [t for t in range(v.arrive, self.inst.horizon - v.handle + 1)]

    def _starts_for(self, i: int, k: int) -> list[int]:
        """Start periods feasible for vessel i at berth k.

        Tidal windows and locks restrict the variable domain, which is tighter
        and smaller than constraints: start_fixed=False keeps every start at or
        after the lower bound, start_fixed=True keeps only the exact start.
        Accepts (bid, start), (bid, start, fixed) and `lock_start_free`.
        """
        inst = self.inst
        v = inst.vessels[i]
        if self.lock and i in self.lock:
            raw = self.lock[i]
            # lock entry is (berth, start lower bound, fixed) -- the berth must
            # be matched against k and the lower bound against the start period.
            lock_berth, lock_lb = raw[0], raw[1]
            fixed = raw[2] if len(raw) > 2 else False
            if lock_berth != k:
                return []
            ok = lambda t: all(                       # noqa: E731
                not inst.blocked(k, tt) for tt in range(t, t + v.handle))
            if self.lock_start_free or not fixed:
                return [t for t in self._starts(i) if t >= lock_lb and ok(t)]
            pinned = [lock_lb] if ok(lock_lb) else []
            if not pinned:
                # The pinned instant was feasible on the planning view but a
                # later tidal shift has blocked it. A commitment to a berth
                # cannot outlast the berth's availability, so keep the berth and
                # relax the start rather than declaring the instance infeasible;
                # a start at or after the commitment still respects the fact that
                # the vessel is already alongside.
                return [t for t in self._starts(i) if t >= lock_lb and ok(t)]
            return pinned
        out = []
        for t in self._starts(i):
            if all(not inst.blocked(k, tt) for tt in range(t, t + v.handle)):
                out.append(t)
        return out

    def build(self):
        inst = self.inst
        n, m, T = inst.n, inst.m, inst.horizon
        self.big_m = safe_big_m(inst)
        prob = pulp.LpProblem(f"DBAP_TI_{inst.name}", pulp.LpMinimize)

        # feasible starts per (i, k)
        starts = {(i, k): self._starts_for(i, k) for i in range(n) for k in range(m)}
        self._starts_cache = starts

        x: dict[tuple[int, int, int], pulp.LpVariable] = {}
        for i in range(n):
            for k in range(m):
                for t in starts[(i, k)]:
                    x[(i, k, t)] = prob.add_variable(f"x_{i}_{k}_{t}", cat="Binary")
        U = {(i, k): prob.add_variable(f"U_{i}_{k}", cat="Binary")
             for i in range(n) for k in range(m)}

        self.problem, self.x, self.U = prob, x, U

        if self.objective == "sum_C":
            prob.setObjective(pulp.lpSum(
                (t + inst.vessels[i].handle) * x[(i, k, t)]
                for (i, k, t) in x
            ))
        elif self.objective == "C_max":
            cmax = prob.add_variable("C_max", lowBound=0)
            for i in range(n):
                terms = [x[(i, k, t)] for k in range(m) for t in starts[(i, k)]]
                if terms:
                    prob.addConstraint(
                        cmax >= pulp.lpSum((t + inst.vessels[i].handle) * x[(i, k, t)]
                                           for k in range(m) for t in starts[(i, k)]),
                        name=f"cmax_{i}")
            prob.setObjective(cmax)
        else:
            raise ValueError(f"unknown objective: {self.objective}")

        for i in range(n):
            terms = [x[(i, k, t)] for k in range(m) for t in starts[(i, k)]]
            if not terms:
                raise RuntimeError(
                    f"vessel {inst.vessels[i].name} has no feasible start "
                    "period at any berth, instance is infeasible")
            prob.addConstraint(pulp.lpSum(terms) == 1, name=f"assign_{i}")

        for i in range(n):
            for k in range(m):
                terms = [x[(i, k, t)] for t in starts[(i, k)]]
                expr = pulp.lpSum(terms) if terms else 0
                prob.addConstraint(expr == U[(i, k)], name=f"defU_{i}_{k}")

        # sum_i [i occupies k at t] * q_i <= L_k,
        # with q_i = L_k if full_berth else length_i
        for k in range(m):
            L_k = inst.berths[k].length
            for t in range(T):
                terms = []
                for i in range(n):
                    v = inst.vessels[i]
                    q = L_k if self.full_berth else v.length
                    sub = [x[(i, k, tp)] for tp in starts[(i, k)]
                           if tp <= t <= tp + v.handle - 1]
                    if sub:
                        terms.append(q * pulp.lpSum(sub))
                if terms:
                    prob.addConstraint(pulp.lpSum(terms) <= L_k, name=f"cap_{k}_{t}")

        # quay crane limit: simultaneous vessels per period, so an outage is a
        # time window rather than a constant
        if inst.quay_crane_limit is not None:
            for t in range(T):
                cap = inst.crane_capacity(t)
                if cap is None:
                    continue
                terms = []
                for i in range(n):
                    v = inst.vessels[i]
                    for k in range(m):
                        sub = [x[(i, k, tp)] for tp in starts[(i, k)]
                               if tp <= t <= tp + v.handle - 1]
                        if sub:
                            terms.append(pulp.lpSum(sub))
                if terms:
                    prob.addConstraint(pulp.lpSum(terms) <= cap, name=f"crane_{t}")
        return prob

    def solve(self) -> Schedule:
        if self.problem is None:
            self.build()
        kwargs: dict = {"msg": self.msg}
        if self.time_limit is not None:
            kwargs["timeLimit"] = self.time_limit
        if self.mip_gap is not None:
            kwargs["gapRel"] = self.mip_gap
        kwargs.update(self.solver_kwargs or {})

        t0 = time.perf_counter()
        stats = self.problem.solve(pulp.HiGHS(**kwargs))
        wall = time.perf_counter() - t0
        status = getattr(stats, "status", None)
        status_str = str(getattr(stats, "status_str", None) or status)
        inst = self.inst

        # A solver that hits a limit can still have a feasible solution in hand.
        # Reporting those as infeasible both discards a usable schedule and
        # misdiagnoses the instance, so classify before deciding.
        solved = status == 1                       # Optimal: proved
        limited = status in LIMIT_STATUSES         # feasible, optimality unproven
        if not solved and not limited:
            return Schedule(assign={}, objective=None, feasible=False,
                            violations=[f"solver returned {status_str}"],
                            solver="MILP-TI", wall_time=wall,
                            extra={"status": status_str, "proved_optimal": False,
                                   "solver_kwargs": dict(kwargs)})

        assign: dict[int, tuple[int, int]] = {}
        for i in range(inst.n):
            hit = None
            for k in range(inst.m):
                for t in self._starts_cache[(i, k)]:
                    val = self.x[(i, k, t)].value() or 0.0
                    if val > 0.5:
                        hit = (k, t)
                        break
                if hit:
                    break
            if hit is None:
                if limited:
                    # limit reached before any integer solution was found
                    return Schedule(assign={}, objective=None, feasible=False,
                                    violations=[f"solver returned {status_str} "
                                                f"before finding a feasible solution"],
                                    solver="MILP-TI", wall_time=wall,
                                    extra={"status": status_str,
                                           "proved_optimal": False,
                                           "solver_kwargs": dict(kwargs)})
                raise RuntimeError(f"no assignment found for vessel V{i}")
            assign[i] = hit

        sch = Schedule(assign=assign, solver="MILP-TI", wall_time=wall,
                       extra={"big_m": self.big_m, "status": status_str,
                              "proved_optimal": solved,
                              "vars": self.problem.numVariables(),
                              "cons": self.problem.numConstraints(),
                              "objective_raw": pulp.value(self.problem.objective)})
        sch.check(inst)
        sch.compute_objective(inst, self.objective)
        return sch


@dataclass
class BAPDisjunctive:
    """Pairwise ordering formulation of min sum C_i.

    Vessels sharing a berth are ordered by a big-M disjunction over their start
    times rather than placed on a time grid, which keeps the ordering variables at
    O(n^2). Berth closures and the crane limit are not expressible in that form
    alone, so this model also carries a period structure: one binary per vessel and
    candidate start period. The result is more variables than the time-indexed
    model and a weaker LP relaxation, which is why it is used as a cross-check
    rather than as the default.
    """
    inst: Instance
    objective: str = "sum_C"
    time_limit: float | None = None
    msg: bool = False

    problem: object = None
    big_m: int = 0

    def build(self):
        inst = self.inst
        n, m, T = inst.n, inst.m, inst.horizon
        M = self.big_m = safe_big_m(inst)
        prob = pulp.LpProblem(f"DBAP_DJ_{inst.name}", pulp.LpMinimize)

        x = {(i, k): prob.add_variable(f"x_{i}_{k}", cat="Binary")
             for i in range(n) for k in range(m)}
        b = {i: prob.add_variable(f"b_{i}", lowBound=inst.vessels[i].arrive,
                                  upBound=inst.horizon, cat="Integer")
             for i in range(n)}
        z = {(i, j): prob.add_variable(f"z_{i}_{j}", cat="Binary")
             for i in range(n) for j in range(i + 1, n)}
        # Candidate start periods per vessel and berth, empty when a closure
        # leaves no unblocked handling window.
        starts = {(i, k): [t for t in range(inst.vessels[i].arrive, T - inst.vessels[i].handle + 1)
                           if not any(inst.blocked(k, tt)
                                      for tt in range(t, t + inst.vessels[i].handle))]
                  for i in range(n) for k in range(m)}
        self._starts = starts
        for i in range(n):
            if not any(starts[(i, k)] for k in range(m)):
                raise RuntimeError(
                    f"vessel {inst.vessels[i].name} has no unblocked handling "
                    f"window on any berth")

        s = {(i, t): prob.add_variable(f"s_{i}_{t}", cat="Binary")
             for i in range(n) for t in range(inst.vessels[i].arrive, T + 1)}
        self.x, self.b, self.z, self.s = x, b, z, s
        self.problem = prob

        if self.objective == "sum_C":
            prob.setObjective(pulp.lpSum(b[i] + inst.vessels[i].handle for i in range(n)))
        else:
            cmax = prob.add_variable("C_max", lowBound=0)
            for i in range(n):
                prob.addConstraint(cmax >= b[i] + inst.vessels[i].handle)
            prob.setObjective(cmax)

        for i in range(n):
            prob.addConstraint(pulp.lpSum(x[(i, k)] for k in range(m)) == 1,
                               name=f"assign_{i}")

        # Link the start period of each vessel to its chosen berth: a start t is
        # only admissible on berths where t is unblocked.
        for i in range(n):
            prob.addConstraint(pulp.lpSum(s[(i, t)] for t in range(inst.vessels[i].arrive,
                                                                   T + 1)) == 1,
                               name=f"start_one_{i}")
            prob.addConstraint(
                b[i] == pulp.lpSum(t * s[(i, t)]
                                   for t in range(inst.vessels[i].arrive, T + 1)),
                name=f"start_of_{i}")
            for k in range(m):
                for t in range(inst.vessels[i].arrive, T + 1):
                    if t not in starts[(i, k)]:
                        prob.addConstraint(
                            s[(i, t)] + x[(i, k)] <= 1,
                            name=f"closed_{i}_{k}_{t}")

        for i in range(n):
            for j in range(i + 1, n):
                pi_, pj_ = inst.vessels[i].handle, inst.vessels[j].handle
                for k in range(m):
                    both = prob.add_variable(f"both_{i}_{j}_{k}", cat="Binary")
                    prob.addConstraint(both <= x[(i, k)])
                    prob.addConstraint(both <= x[(j, k)])
                    prob.addConstraint(both >= x[(i, k)] + x[(j, k)] - 1)
                    prob.addConstraint(b[i] + pi_ <= b[j] + M * (1 - z[(i, j)]) + M * (1 - both),
                                       name=f"seqA_{i}_{j}_{k}")
                    prob.addConstraint(b[j] + pj_ <= b[i] + M * z[(i, j)] + M * (1 - both),
                                       name=f"seqB_{i}_{j}_{k}")

        # Crane limit: the number of vessels handling at any period may not exceed
        # the cranes available then. Occupancy per berth and period is the sum of
        # the vessels whose chosen berth is k, whose start is at or before t, and
        # whose handling has not finished by t.
        if inst.quay_crane_limit is not None:
            for t in range(T):
                occupancy = []
                for i in range(n):
                    span = range(max(inst.vessels[i].arrive, t - inst.vessels[i].handle + 1),
                                 t + 1)
                    for k in range(m):
                        if not any(tt in starts[(i, k)] for tt in span):
                            continue
                        for tp in span:
                            if tp not in starts[(i, k)]:
                                continue
                            occ = prob.add_variable(f"occ_{i}_{k}_{t}_{tp}", cat="Binary")
                            prob.addConstraint(occ <= x[(i, k)])
                            prob.addConstraint(occ <= s[(i, tp)])
                            prob.addConstraint(
                                occ >= x[(i, k)] + s[(i, tp)] - 1)
                            occupancy.append(occ)
                cap = inst.crane_capacity(t)
                if occupancy and cap is not None:
                    prob.addConstraint(pulp.lpSum(occupancy) <= cap,
                                       name=f"crane_{t}")
        return prob

    def solve(self) -> Schedule:
        if self.problem is None:
            self.build()
        kwargs = {"msg": self.msg}
        if self.time_limit is not None:
            kwargs["timeLimit"] = self.time_limit
        t0 = time.perf_counter()
        stats = self.problem.solve(pulp.HiGHS(**kwargs))
        wall = time.perf_counter() - t0
        status = getattr(stats, "status", None)
        status_str = str(getattr(stats, "status_str", None) or status)
        inst = self.inst
        solved = status == 1
        limited = status in LIMIT_STATUSES
        if not solved and not limited:
            return Schedule(assign={}, objective=None, feasible=False,
                            violations=[f"solver returned {status_str}"],
                            solver="MILP-DJ", wall_time=wall,
                            extra={"status": status_str, "proved_optimal": False})
        assign = {}
        try:
            for i in range(inst.n):
                k = max(range(inst.m), key=lambda kk: self.x[(i, kk)].value() or 0.0)
                assign[i] = (k, int(round(self.b[i].value())))
        except (TypeError, ValueError):
            # a limit can be reached before an integer solution exists
            return Schedule(assign={}, objective=None, feasible=False,
                            violations=[f"solver returned {status_str} before "
                                        f"finding a feasible solution"],
                            solver="MILP-DJ", wall_time=wall,
                            extra={"status": status_str, "proved_optimal": False})
        sch = Schedule(assign=assign, solver="MILP-DJ", wall_time=wall,
                       extra={"vars": self.problem.numVariables(),
                              "cons": self.problem.numConstraints(),
                              "status": status_str, "proved_optimal": solved})
        sch.check(inst)
        sch.compute_objective(inst, self.objective)
        return sch


def solve_bap(inst: Instance, model: str = "ti", objective: str = "sum_C",
              time_limit: float | None = None, msg: bool = False) -> Schedule:
    """Convenience entry point. model: 'ti' = time-indexed, 'dj' = disjunctive."""
    if model == "ti":
        return BAPTimeIndexed(inst, objective=objective,
                              time_limit=time_limit, msg=msg).solve()
    if model == "dj":
        return BAPDisjunctive(inst, objective=objective,
                              time_limit=time_limit, msg=msg).solve()
    raise ValueError(f"unknown model: {model}")
