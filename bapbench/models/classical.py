# -*- coding: utf-8 -*-
"""Classical scheduling models.

Three-field notation alpha | beta | gamma (Graham et al. 1979): alpha is the
machine environment, beta the job and machine characteristics, gamma the
objective. Targets covered here:

    1| |sum C_j            single machine, total completion time (SPT optimal)
    1|r_j|sum C_j          with release dates (NP-hard)
    P| |C_max              parallel machines, makespan (NP-hard, LPT has 4/3 bound)
    P|r_j|sum C_j          with release dates (NP-hard)
    P|r_j, M_j|sum C_j     plus machine availability windows
    F|prmu|C_max           permutation flow shop (NP-hard)
    J| |C_max              job shop, the general case (NP-hard)

Port analogue: berth = machine, vessel = job, arrival = release date, tidal
window = machine availability, vessel length = capacity requirement.

Each model has-a SimpleProblem and exposes .build(), .solve() returning a
Schedule so metrics match the BAP models, and .formulation() returning
(variables, constraints, binaries).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import permutations

import pulp

from ..instances import Instance, Schedule, Vessel, Berth



# Problem definition

@dataclass
class Job:
    jid: int
    name: str
    release: int = 0            # r_j
    proc: int = 1               # p_j
    weight: int = 1             # w_j


@dataclass
class Machine:
    mid: int
    name: str
    unavailable: list[tuple[int, int]] = field(default_factory=list)   # M_j

    def blocked(self, t: int) -> bool:
        return any(a <= t < b for (a, b) in self.unavailable)


@dataclass
class SimpleProblem:
    """Common input for the classical models."""
    name: str
    jobs: list[Job]
    machines: list[Machine]
    horizon: int

    @property
    def n(self) -> int:
        return len(self.jobs)

    @property
    def m(self) -> int:
        return len(self.machines)

    def summary(self) -> str:
        return (f"{self.name}: {self.n} jobs x {self.m} machines | "
                f"horizon {self.horizon}")


def make_simple_problem(name: str, n: int, m: int, *,
                        proc_range=(2, 8), horizon: int | None = None,
                        release: bool = False, release_range=(0, 10),
                        unavailable: bool = False, seed: int = 0) -> SimpleProblem:
    """Build a classical scheduling instance."""
    import random
    rng = random.Random(seed)
    jobs = []
    for j in range(n):
        jobs.append(Job(
            jid=j, name=f"J{j+1}",
            release=rng.randint(*release_range) if release else 0,
            proc=rng.randint(*proc_range),
            weight=rng.randint(1, 3),
        ))
    machines = []
    for k in range(m):
        un = []
        if unavailable:
            period, block = 12, 4
            phase = k * (period // max(1, m))
            for a in range(phase, (horizon or 999), period):
                un.append((a, min(a + block, horizon or 999)))
        machines.append(Machine(mid=k, name=f"M{k+1}", unavailable=un))

    if horizon is None:
        horizon = (sum(j.proc for j in jobs) + max(j.release for j in jobs)
                   + (12 if unavailable else 0) + 5)

    return SimpleProblem(name=name, jobs=jobs, machines=machines, horizon=horizon)


def to_instance_view(P: SimpleProblem) -> Instance:
    """Adapt a SimpleProblem into the package's Instance so classical models can
    reuse the same feasibility checker and metrics.

    Jobs map to vessels, machines to berths, machine unavailability to tidal
    windows, and release dates to arrival times. Capacity is one job per machine
    at a time: berths are given full-length vessels, so the quay capacity check
    reduces to non-overlap.
    """
    vessels = [
        Vessel(vid=j.jid, name=j.name, length=1, arrive=j.release,
               handle=j.proc, priority=j.weight)
        for j in P.jobs
    ]
    berths = [Berth(bid=k.mid, name=k.name, length=1) for k in P.machines]
    tidal = {str(k.mid): list(k.unavailable) for k in P.machines if k.unavailable}
    inst = Instance(
        name=P.name, vessels=vessels, berths=berths, horizon=P.horizon,
        tidal_windows=tidal,
        quay_crane_limit=None,
        meta={"adapter": "w3.to_instance_view"},
    )
    return inst



# Results reuse the BAP Schedule so metrics stay comparable

def _to_schedule(inst_like, start: dict[int, int], machine_of: dict[int, int],
                 prob: SimpleProblem, solver: str, wall: float,
                 nvars: int, ncons: int, nbin: int,
                 objective: str = "sum_C") -> Schedule:
    """Wrap a classical solution as a Schedule to reuse the same metrics.

    The objective must be reported in the units the model minimised: summing the
    completion times gives a different number from their maximum, so reporting the
    wrong one silently mislabels the result.
    """
    assign = {j: (machine_of[j], start[j]) for j in range(prob.n)}
    sch = Schedule(assign=assign, solver=solver, wall_time=wall,
                   extra={"vars": nvars, "cons": ncons, "binaries": nbin})
    sch.check(inst_like)
    comp = sch.completion(inst_like)
    if objective == "sum_C":
        sch.objective = float(sum(comp.values()))
    elif objective == "C_max":
        sch.objective = float(max(comp.values()))
    else:
        raise ValueError(f"unknown objective: {objective}")
    return sch


def makespan_upper_bound(P: SimpleProblem) -> int:
    """Safe makespan upper bound, used to bound C[j][k] and C_max.

    Processing all jobs serially on every machine gives

        C_max <= max_j r_j + m * sum_j p_j

    Loose but always attainable, hence safe as a variable bound: a bound only has
    to leave the optimum feasible. The single-machine form
    max_j r_j + sum_j p_j is not valid for m > 1, because a job still occupies
    each of the m machines in turn.
    """
    return (max((j.release for j in P.jobs), default=0)
            + P.m * sum(j.proc for j in P.jobs))


def _require_no_release_dates(P: SimpleProblem, model: str) -> None:
    """Reject an instance this model cannot represent.

    The flow-shop and job-shop formulations start every job at time zero and do
    not carry release dates, so an instance with them would be solved as a
    different problem and returned as infeasible (the adapter's feasibility check
    sees starts before arrivals). Failing loudly beats returning a schedule for
    the wrong problem.
    """
    if any(j.release for j in P.jobs):
        raise ValueError(
            f"{model} does not support release dates, but this instance has them "
            f"(min {min(j.release for j in P.jobs)}, "
            f"max {max(j.release for j in P.jobs)}); use ParallelMachineMILP, "
            f"which carries r_j, or clear the releases")


# 1| |sum C_j: SPT rule, polynomial, no MILP needed

def single_machine_spt(prob: SimpleProblem) -> tuple[list[int], float]:
    """SPT: ascending p_j, optimal for 1| |sum C_j. Returns (order, sum C_j).

    Release dates are honoured, so the returned value is the cost of a feasible
    schedule. With every release date zero this is the proven optimum; once
    release dates are present SPT is a heuristic only, since 1|r_j|sum C_j is
    NP-hard.
    """
    order = sorted(range(prob.n), key=lambda j: (prob.jobs[j].proc, j))
    t, total = 0, 0.0
    for j in order:
        t = max(t, prob.jobs[j].release) + prob.jobs[j].proc
        total += t
    return order, total


def single_machine_wspt(prob: SimpleProblem) -> tuple[list[int], float]:
    """WSPT: ascending p_j / w_j, optimal for 1| |sum w_j C_j. Returns (order, sum w_j C_j).

    Release dates are honoured, so the returned value is the cost of a feasible
    schedule. With every release date zero this is the proven optimum; with
    release dates it is a heuristic.
    """
    from fractions import Fraction
    order = sorted(range(prob.n),
                   key=lambda j: (Fraction(prob.jobs[j].proc, prob.jobs[j].weight), j))
    t, total = 0, 0.0
    for j in order:
        t = max(t, prob.jobs[j].release) + prob.jobs[j].proc
        total += prob.jobs[j].weight * t
    return order, total



# MILP: single machine / parallel machines with release dates and availability

@dataclass
class ParallelMachineMILP:
    """P | r_j, M_j | sum C_j as a time-indexed assignment model.

    Variables:
        x[j,k,t] = 1  job j starts on machine k at time t
    Constraints:
        (1) sum_{k,t} x[j,k,t] = 1              each job exactly once
        (2) at most one job per machine at any time
        (3) infeasible (k,t) pairs get no variable, tighter than big-M
    Objective: sum C_j or C_max
    """
    prob: SimpleProblem
    objective: str = "sum_C"
    time_limit: float | None = None
    msg: bool = False
    instance_view: Instance | None = None

    problem: object = None
    x: dict = None
    _starts: dict = None

    def _feasible_starts(self, j: int, k: int) -> list[int]:
        job = self.prob.jobs[j]
        mach = self.prob.machines[k]
        out = []
        for t in range(job.release, self.prob.horizon - job.proc + 1):
            if any(mach.blocked(tt) for tt in range(t, t + job.proc)):
                continue
            out.append(t)
        return out

    def build(self):
        P = self.prob
        n, m, T = P.n, P.m, P.horizon
        model = pulp.LpProblem(f"PM_{P.name}", pulp.LpMinimize)

        starts = {(j, k): self._feasible_starts(j, k) for j in range(n) for k in range(m)}
        self._starts = starts

        x = {}
        for j in range(n):
            for k in range(m):
                for t in starts[(j, k)]:
                    x[(j, k, t)] = model.add_variable(f"x_{j}_{k}_{t}", cat="Binary")
        self.problem, self.x = model, x

        if self.objective == "sum_C":
            model.setObjective(pulp.lpSum(
                (t + P.jobs[j].proc) * x[(j, k, t)] for (j, k, t) in x))
        elif self.objective == "C_max":
            cmax = model.add_variable("C_max", lowBound=0)
            for j in range(n):
                terms = [x[(j, k, t)] for k in range(m) for t in starts[(j, k)]]
                if terms:
                    model.addConstraint(
                        cmax >= pulp.lpSum((t + P.jobs[j].proc) * x[(j, k, t)]
                                           for k in range(m) for t in starts[(j, k)]),
                        name=f"cmax_{j}")
            model.setObjective(cmax)
        else:
            raise ValueError(self.objective)

        # (1) each job exactly once
        for j in range(n):
            terms = [x[(j, k, t)] for k in range(m) for t in starts[(j, k)]]
            if not terms:
                raise RuntimeError(f"job {P.jobs[j].name} has no feasible start, "
                                   f"instance infeasible")
            model.addConstraint(pulp.lpSum(terms) == 1, name=f"assign_{j}")

        # (2) at most one job per machine at any time
        for k in range(m):
            for t in range(T):
                terms = []
                for j in range(n):
                    sub = [x[(j, k, tp)] for tp in starts[(j, k)]
                           if tp <= t <= tp + P.jobs[j].proc - 1]
                    if sub:
                        terms.append(pulp.lpSum(sub))
                if terms:
                    model.addConstraint(pulp.lpSum(terms) <= 1, name=f"cap_{k}_{t}")
        return model

    def formulation(self) -> tuple[int, int, int]:
        model = self.problem
        nbin = sum(1 for v in model.variables() if v.cat == "Binary")
        return model.numVariables(), model.numConstraints(), nbin

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
        nv, nc, nb = self.formulation()
        if status != 1:
            return Schedule(assign={}, feasible=False, objective=None,
                            violations=[f"no optimal solution status={status}"],
                            solver="PM-MILP", wall_time=wall,
                            extra={"vars": nv, "cons": nc, "binaries": nb})
        start, mach_of = {}, {}
        for j in range(self.prob.n):
            hit = None
            for k in range(self.prob.m):
                for t in self._starts[(j, k)]:
                    if (self.x[(j, k, t)].value() or 0) > 0.5:
                        hit = (k, t)
                        break
                if hit:
                    break
            if hit is None:
                raise RuntimeError(f"no assignment read for job J{j}")
            mach_of[j], start[j] = hit
        return _to_schedule(self.instance_view, start, mach_of, self.prob,
                            "PM-MILP", wall, nv, nc, nb,
                            objective=self.objective)



# Flow shop F|prmu|C_max

@dataclass
class FlowShopMILP:
    """F | prmu | C_max: the permutation flow shop, where all jobs visit the
    machines in the same order.

    Variables: binary y[j][j'] for job j preceding job j'.
    Constraints: pairwise ordering on each machine, plus precedence of each job
    across machines.
    """
    prob: SimpleProblem
    time_limit: float | None = None
    msg: bool = False
    instance_view: Instance | None = None

    problem: object = None

    def build(self):
        P = self.prob
        n, m, T = P.n, P.m, P.horizon
        _require_no_release_dates(P, "FlowShopMILP")
        model = pulp.LpProblem(f"FS_{P.name}", pulp.LpMinimize)
        UB = makespan_upper_bound(P)
        M = UB + 1

        # C[j][k] = completion time of job j on machine k
        C = {(j, k): model.add_variable(f"C_{j}_{k}", lowBound=0, upBound=UB, cat="Integer")
             for j in range(n) for k in range(m)}
        cmax = model.add_variable("C_max", lowBound=0, upBound=UB)
        y = {(j, jp): model.add_variable(f"y_{j}_{jp}", cat="Binary")
             for j in range(n) for jp in range(n) if j != jp}
        self.problem, self.C, self.y = model, C, y
        self.ub = UB
        model.setObjective(cmax)

        for j in range(n):
            p0 = P.jobs[j].proc
            model.addConstraint(C[(j, 0)] >= p0, name=f"first_{j}")
            for k in range(1, m):
                model.addConstraint(C[(j, k)] >= C[(j, k - 1)] + P.jobs[j].proc,
                                    name=f"prec_{j}_{k}")
            model.addConstraint(cmax >= C[(j, m - 1)], name=f"cmax_{j}")

        # Disjunction on each machine. y[j,j'] = 1 means j' follows j, so the
        # constraint must carry p_{j'}, not p_j:
        #     C[j'] >= C[j] + p_{j'}  <=>  C[j] + p_{j'} <= C[j'] + M(1-y[j,j'])
        # Using p_j contradicts the ordering whenever p_j > p_{j'}.
        for k in range(m):
            for j in range(n):
                for jp in range(n):
                    if j == jp:
                        continue
                    model.addConstraint(
                        C[(j, k)] + P.jobs[jp].proc
                        <= C[(jp, k)] + M * (1 - y[(j, jp)]),
                        name=f"disj_{j}_{jp}_{k}")
        # Antisymmetry: y[j,j'] + y[j',j] = 1
        for j in range(n):
            for jp in range(j + 1, n):
                model.addConstraint(y[(j, jp)] + y[(jp, j)] == 1, name=f"antisym_{j}_{jp}")
        return model

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
        nv, nc = self.problem.numVariables(), self.problem.numConstraints()
        if status != 1:
            return Schedule(assign={}, feasible=False, objective=None,
                            violations=[f"no optimal solution status={status}"],
                            solver="FS-MILP", wall_time=wall,
                            extra={"vars": nv, "cons": nc})
        # Recover the order from each job's start on the first machine:
        # start_j = C[j][0] - p_j
        start = {j: int(round(self.C[(j, 0)].value())) - self.prob.jobs[j].proc
                 for j in range(self.prob.n)}
        mach_of = {j: 0 for j in range(self.prob.n)}
        sch = _to_schedule(self.instance_view, start, mach_of, self.prob,
                           "FS-MILP", wall, nv, nc, self.prob.n * self.prob.n)
        sch.objective = float(round(self.problem.objective.value(), 6))
        return sch



# Job shop J| |C_max

@dataclass
class JobShopMILP:
    """J | |C_max: job shop, where each job has a fixed machine route.

    Every operation of a job is assumed to have the same duration (Job.proc).
    Varying per-operation durations need the formulation in
    verification/benchmarks.py.
    """
    prob: SimpleProblem
    routes: dict[int, list[int]] = None       # j -> machine sequence
    time_limit: float | None = None
    msg: bool = False
    instance_view: Instance | None = None

    problem: object = None

    def build(self):
        P = self.prob
        n, m, T = P.n, P.m, P.horizon
        _require_no_release_dates(P, "JobShopMILP")
        if self.routes is None:
            # Default route: 0..m-1 for every job
            self.routes = {j: list(range(m)) for j in range(n)}

        model = pulp.LpProblem(f"JS_{P.name}", pulp.LpMinimize)
        UB = makespan_upper_bound(P)
        M = UB + 1
        # s[j][k] = start time of job j's k-th operation
        s = {(j, k): model.add_variable(f"s_{j}_{k}", lowBound=0, upBound=UB, cat="Integer")
             for j in range(n) for k in range(len(self.routes[j]))}
        cmax = model.add_variable("C_max", lowBound=0, upBound=UB)
        self.problem, self.s = model, s
        self.ub = UB
        model.setObjective(cmax)

        for j in range(n):
            route = self.routes[j]
            for k in range(1, len(route)):
                model.addConstraint(s[(j, k)] >= s[(j, k - 1)] + P.jobs[j].proc,
                                    name=f"route_{j}_{k}")
            last = len(route) - 1
            model.addConstraint(cmax >= s[(j, last)] + P.jobs[j].proc, name=f"cmax_{j}")

        # Disjunction between every pair of operations sharing a machine
        ops = []          # (j, k, machine)
        for j in range(n):
            for k, mach in enumerate(self.routes[j]):
                ops.append((j, k, mach))
        z = {}
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                j, k, mk = ops[a]
                jp, kp, mkp = ops[b]
                if mk != mkp or j == jp:
                    continue
                zz = model.add_variable(f"z_{j}_{k}_{jp}_{kp}", cat="Binary")
                z[(j, k, jp, kp)] = zz
                model.addConstraint(
                    s[(j, k)] + P.jobs[j].proc <= s[(jp, kp)] + M * (1 - zz),
                    name=f"jsA_{j}_{k}_{jp}_{kp}")
                model.addConstraint(
                    s[(jp, kp)] + P.jobs[jp].proc <= s[(j, k)] + M * zz,
                    name=f"jsB_{j}_{k}_{jp}_{kp}")
        self.z = z
        return model

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
        nv, nc = self.problem.numVariables(), self.problem.numConstraints()
        if status != 1:
            return Schedule(assign={}, feasible=False, objective=None,
                            violations=[f"no optimal solution status={status}"],
                            solver="JS-MILP", wall_time=wall,
                            extra={"vars": nv, "cons": nc})
        start = {j: int(round(self.s[(j, 0)].value())) for j in range(self.prob.n)}
        mach_of = {j: self.routes[j][0] for j in range(self.prob.n)}
        sch = _to_schedule(self.instance_view, start, mach_of, self.prob,
                           "JS-MILP", wall, nv, nc, len(self.z))
        sch.objective = float(round(self.problem.objective.value(), 6))
        return sch



# Exhaustive bounds for validating the MILPs

def brute_force_single_machine(prob: SimpleProblem,
                               objective: str = "sum_C") -> tuple[float | None, list[int] | None]:
    """Exhaustive single-machine search with release dates. Returns (value, order)."""
    n = prob.n
    if n > 8:
        return None, None
    best, best_order = None, None
    for order in permutations(range(n)):
        t, total = 0, 0.0
        ok = True
        for j in order:
            job = prob.jobs[j]
            t = max(t, job.release)
            t += job.proc
            total += t
            if t > prob.horizon:
                ok = False
                break
        if not ok:
            continue
        val = total if objective == "sum_C" else t
        if best is None or val < best:
            best, best_order = val, list(order)
    return best, best_order


def parallel_search_work(n: int, m: int, horizon: int,
                         blocked_fraction: float = 0.0) -> int:
    """Number of complete schedules ``brute_force_parallel`` would evaluate.

    Every assignment of n distinguishable jobs to m machines is followed by every
    ordering on each machine. Grouping assignments by the machine load vector
    (k_0 .. k_{m-1}) sums to

        sum over sum(k_i) = n of  n!/prod(k_i!) * prod(k_i!)  =  n! * C(n+m-1, m-1)

    since the multinomial factor and the orderings cancel. Counting only ``m**n``
    assignments understates this badly and grows apart from it: at n=10, m=3 the
    assignment count is 59,049 while the real enumeration is 9,239,040.

    The blocking test walks one period at a time on an unavailable machine, which
    multiplies each evaluated schedule by roughly ``1 + blocked_fraction``.
    """
    from math import comb, factorial

    schedules = factorial(n) * comb(n + m - 1, m - 1)
    return int(schedules * (1.0 + max(0.0, blocked_fraction)))


def brute_force_parallel(prob: SimpleProblem, objective: str = "sum_C",
                         max_cases: int = 2_000_000) -> float | None:
    """Exhaustive parallel-machine search over assignments and per-machine orders.

    Returns None when the enumeration is estimated to exceed ``max_cases``, so the
    caller can tell "too large to enumerate" from "no feasible schedule".
    """
    n, m = prob.n, prob.m
    blocked = 0.0
    if prob.horizon and prob.machines:
        covered = sum(1 for k in prob.machines
                      for t in range(prob.horizon) if k.blocked(t))
        blocked = covered / (len(prob.machines) * prob.horizon)
    if parallel_search_work(n, m, prob.horizon, blocked) > max_cases:
        return None
    best = None
    for combo in __import__("itertools").product(range(m), repeat=n):
        groups = {k: [] for k in range(m)}
        for j in range(n):
            groups[combo[j]].append(j)
        permsets = [list(permutations(g)) for g in groups.values()]
        for combo_perms in __import__("itertools").product(*permsets):
            starts = {}
            ok = True
            for k, p in zip(groups.keys(), combo_perms):
                free = 0
                for j in p:
                    job = prob.jobs[j]
                    t = max(job.release, free)
                    # Step over unavailable machine intervals
                    while any(prob.machines[k].blocked(tt) for tt in range(t, t + job.proc)):
                        t += 1
                        if t + job.proc > prob.horizon:
                            ok = False
                            break
                    if not ok:
                        break
                    if t + job.proc > prob.horizon:
                        ok = False
                        break
                    starts[j] = t
                    free = t + job.proc
                if not ok:
                    break
            if not ok:
                continue
            comp = {j: starts[j] + prob.jobs[j].proc for j in starts}
            if len(comp) != n:
                continue
            val = float(sum(comp.values())) if objective == "sum_C" else float(max(comp.values()))
            if best is None or val < best:
                best = val
    return best
