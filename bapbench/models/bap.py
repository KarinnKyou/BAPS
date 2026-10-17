# -*- coding: utf-8 -*-
"""Exact MILP formulations for the dynamic berth allocation problem.

Imai, Nishimura & Papadimitriou (2001), Transportation Research Part B 35(4).
`sum_i C_i` is the default objective; `C_max` is also supported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pulp

from ..instances import Instance, Schedule
from ..metrics import LIMIT_STATUSES, classify_status  # noqa: F401  (LIMIT_STATUSES is re-exported)


def safe_big_m(inst: Instance) -> int:
    """Big-M for the pairwise ordering constraints.

    Too small a value does not make the solver report infeasibility; it returns
    a plausible wrong answer. Computed per instance as `horizon + 1`.
    """
    return inst.horizon + 1


def stranded_vessels(inst: Instance,
                     starts: Callable[[int, int], list[int]]) -> list[int]:
    """Vessels with no feasible start at any berth, which makes the instance
    infeasible before any solver is involved.

    A vessel is stranded when a closure covers every one of its windows, when it
    is longer than every berth, or both. Reporting that as an infeasible schedule
    rather than raising keeps a caller that loops over instances from having to
    catch, and lets `violations` say which case it was.
    """
    return [i for i in range(inst.n)
            if not any(starts(i, k) for k in range(inst.m))]


def infeasible_schedule(inst: Instance, stranded: list[int],
                        solver: str) -> Schedule:
    names = [inst.vessels[i].name for i in stranded]
    return Schedule(
        assign={}, feasible=False, objective=None, solver=solver,
        violations=[f"{len(stranded)} vessels can berth at no berth: {names[:4]}"],
        extra={"status": "Infeasible", "proved_optimal": False,
               "has_solution": False})


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
    time_limit: float | None = None
    mip_gap: float | None = None
    msg: bool = False
    solver_kwargs: dict = field(default_factory=dict)
    # {vid: (berth_id, start_lower_bound, start_fixed)}. A port can commit to a
    # berth, but a start can only slip later, never earlier: start_fixed=False
    # postpones the start, start_fixed=True pins berth and start.
    lock: dict = field(default_factory=dict)
    lock_start_free: bool = False     # read every lock as berth-only
    # Drop the integrality restrictions and solve the LP relaxation. Its optimum
    # is a valid lower bound for the integer problem and is used as a reference
    # where no proved optimum exists.
    relax: bool = False

    problem: object = None
    x: dict = None
    U: dict = None
    big_m: int = 0

    def _starts(self, i: int) -> list[int]:
        v = self.inst.vessels[i]
        return [t for t in range(v.arrive, self.inst.horizon - v.handle + 1)]

    def _fits(self, i: int, k: int) -> bool:
        """Whether vessel i is short enough to berth at berth k.

        A vessel occupies the whole berth, so a vessel longer than the berth has no
        feasible start there. `Instance.validate`, `Schedule.check` and
        `metrics.sanity_check_schedule` all enforce this, so the model must too;
        otherwise its proved optimum can be a schedule the package's own check
        rejects, and that value becomes the reference for every reported gap.
        """
        return self.inst.vessels[i].length <= self.inst.berths[k].length

    def _starts_for(self, i: int, k: int) -> list[int]:
        """Start periods feasible for vessel i at berth k.

        Tidal windows and locks restrict the variable domain, which is tighter
        and smaller than constraints: start_fixed=False keeps every start at or
        after the lower bound, start_fixed=True keeps only the exact start.
        Accepts (bid, start), (bid, start, fixed) and `lock_start_free`.
        """
        inst = self.inst
        v = inst.vessels[i]
        if not self._fits(i, k):
            return []
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
            if self.lock_start_free:
                # Every start is allowed: the berth commitment is kept but the
                # start carries no lower bound, which is what the flag means. With
                # the bound still applied the fallback in `rolling_horizon` could
                # not recover a commitment whose start a disruption invalidated.
                return [t for t in self._starts(i) if ok(t)]
            if not fixed:
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
        if not self._fits(i, k):
            # A vessel occupies the whole berth, so it can only berth where its
            # length fits. Leaving this out lets the model report an optimum that
            # the package's own feasibility check rejects, which then becomes the
            # reference every gap is measured against.
            return out
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
        # A Binary variable is implicitly bounded to [0, 1]; a Continuous one is
        # not, so the relaxation must state the bounds or the LP is unbounded.
        if self.relax:
            bounds = {"lowBound": 0, "upBound": 1, "cat": "Continuous"}
        else:
            bounds = {"cat": "Binary"}
        for i in range(n):
            for k in range(m):
                for t in starts[(i, k)]:
                    x[(i, k, t)] = prob.add_variable(f"x_{i}_{k}_{t}", **bounds)
        # No per-(vessel, berth) indicator variables: `assign_i` below already
        # forces exactly one start per vessel, and the per-period rows enforce
        # capacity. Such an indicator was carried but never consulted, and
        # because nothing bounded it from below it made the LP relaxation
        # unbounded.
        self.problem, self.x, self.U = prob, x, {}

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
            prob.addConstraint(pulp.lpSum(terms) == 1, name=f"assign_{i}")

        # No U variables: nothing consults them. Assignment is enforced by
        # `assign_i` below and capacity by the per-period rows, so an indicator
        # per (vessel, berth) would only add free variables -- and would make the
        # LP relaxation unbounded, since nothing bounds them from below.
        self.problem, self.x, self.U = prob, x, {}
        # A vessel occupies its whole berth, so one vessel per berth-period is
        # both necessary and sufficient for quay capacity. Summing vessel lengths
        # instead would admit two short vessels in one berth at the same time,
        # which this model has no position variables to express.
        for k in range(m):
            for t in range(T):
                terms = []
                for i in range(n):
                    v = inst.vessels[i]
                    sub = [x[(i, k, tp)] for tp in starts[(i, k)]
                           if tp <= t <= tp + v.handle - 1]
                    if sub:
                        terms.append(pulp.lpSum(sub))
                if terms:
                    prob.addConstraint(pulp.lpSum(terms) <= 1, name=f"cap_{k}_{t}")

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
        # A vessel that can berth nowhere makes the instance infeasible before the
        # solver runs, so report that as a schedule rather than raising.
        stranded = stranded_vessels(self.inst, self._starts_for)
        if stranded:
            return infeasible_schedule(self.inst, stranded, "MILP")
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
        info = classify_status(stats)
        status_str = info["status"]
        inst = self.inst

        # A solver that hits a limit can still have a feasible solution in hand.
        # Reporting those as infeasible both discards a usable schedule and
        # misdiagnoses the instance, so classify before deciding.
        solved = info["solved"]
        limited = info["limited"]
        # A stopped solver still leaves column values behind, and reading them
        # yields an assignment that is not feasible anywhere.
        has_solution = info["has_solution"]

        if self.relax:
            # The relaxation has no integer assignment to read; it exists to
            # supply a bound. Only an optimally solved LP yields one.
            value = (pulp.value(self.problem.objective)
                     if solved and self.problem.objective is not None else None)
            return Schedule(assign={}, objective=value,
                            feasible=False, violations=[],
                            solver="LP-TI", wall_time=wall,
                            extra={"status": status_str, "is_relaxation": True,
                                   "proved_optimal": solved,
                                   "vars": self.problem.numVariables(),
                                   "cons": self.problem.numConstraints()})

        if (not solved and not limited) or not has_solution:
            return Schedule(assign={}, objective=None, feasible=False,
                            violations=[f"solver returned {status_str}"
                                        + ("" if has_solution else " with no solution")],
                            solver="MILP-TI", wall_time=wall,
                            extra={"status": status_str, "proved_optimal": False,
                                   "has_solution": has_solution,
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
    solver_kwargs: dict = field(default_factory=dict)

    problem: object = None
    big_m: int = 0

    def _starts_for(self, i: int, k: int) -> list[int]:
        """Start periods feasible for vessel i at berth k.

        Empty when a closure covers every window or when the vessel is longer than
        the berth, which is the same condition the time-indexed model applies and
        the same one the package's feasibility checks enforce. This model carries no
        commitments: it is a cross-check on the exact optimum, not a planner.
        """
        if self.inst.vessels[i].length > self.inst.berths[k].length:
            return []
        return self._unblocked(i, k)

    def _unblocked(self, i: int, k: int) -> list[int]:
        inst = self.inst
        vessel = inst.vessels[i]
        last = inst.horizon - vessel.handle
        return [t for t in range(vessel.arrive, last + 1) if self._free(i, k, t)]

    def _free(self, i: int, k: int, t: int) -> bool:
        vessel = self.inst.vessels[i]
        return all(not self.inst.blocked(k, tt)
                   for tt in range(t, t + vessel.handle))

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
        # Candidate start periods per vessel and berth, from the one function that
        # decides them: empty when a closure leaves no unblocked window, when the
        # vessel is longer than the berth, or when a lock excludes the berth.
        starts = {(i, k): self._starts_for(i, k)
                  for i in range(n) for k in range(m)}
        self._starts = starts

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
                if pi_ <= 0 or pj_ <= 0:
                    # A vessel with no handling time occupies no period, so it
                    # cannot be ordered against another. The disjunction would
                    # otherwise force it entirely before or entirely after, which
                    # excludes schedules the instance admits.
                    continue
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
        # A vessel that can berth nowhere makes the instance infeasible before the
        # solver runs, so report that as a schedule rather than raising.
        stranded = stranded_vessels(self.inst, self._starts_for)
        if stranded:
            return infeasible_schedule(self.inst, stranded, "MILP")
        if self.problem is None:
            self.build()
        kwargs = {"msg": self.msg}
        if self.time_limit is not None:
            kwargs["timeLimit"] = self.time_limit
        # Presolve must be off for this formulation. With HiGHS 1.15.1's default
        # presolve the model can return a suboptimal objective while reporting
        # status Optimal and a schedule its own check() accepts: on a 4-vessel,
        # 3-berth instance with a crane limit it returned 44 where the
        # time-indexed model, this model with presolve off, and an exhaustive
        # search all return 42. Solving the same matrix written to an LP file also
        # gives 42, so the fault is in the in-memory presolve path. Turning it off
        # costs a little time and restores agreement.
        kwargs.setdefault("presolve", "off")
        kwargs.update(self.solver_kwargs or {})
        t0 = time.perf_counter()
        stats = self.problem.solve(pulp.HiGHS(**kwargs))
        wall = time.perf_counter() - t0
        info = classify_status(stats)
        status_str = info["status"]
        inst = self.inst
        solved, limited = info["solved"], info["limited"]
        # A stopped solver still leaves column values behind, and reading them
        # yields an assignment that is not feasible anywhere. `has_solution` is
        # the solver's own statement that a solution exists, so gate on it rather
        # than trusting whatever the variables happen to hold.
        has_solution = info["has_solution"]
        if (not solved and not limited) or not has_solution:
            return Schedule(assign={}, objective=None, feasible=False,
                            violations=[f"solver returned {status_str}"
                                        + ("" if has_solution else " with no solution")],
                            solver="MILP-DJ", wall_time=wall,
                            extra={"status": status_str, "proved_optimal": False,
                                   "has_solution": has_solution})
        assign = {}
        try:
            for i in range(inst.n):
                k = max(range(inst.m), key=lambda kk: self.x[(i, kk)].value() or 0.0)
                assign[i] = (k, int(round(self.b[i].value())))
        except (TypeError, ValueError):
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
