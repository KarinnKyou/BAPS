"""Run the reproduced published methods beside the package's own baselines.

Each reproduction solves a slightly different problem from the package's DBAP, so
this module fixes what is compared and what is not.  Every method is scored on the
same objective, computed from the assignment alone by `metrics.objective_value`,
and every schedule is re-checked with `metrics.sanity_check_schedule`, which shares
no code with any of them.  A method that cannot place every vessel is reported as
a failure with its reason rather than dropped.

* Iris 2017 solves the integrated berth allocation and quay crane assignment
  problem on the package's own instance type, so its result is directly comparable.
* Imai 2001 solves the DBAP, whose handling time depends on the berth and whose
  objective is total waiting plus handling.  On a package instance the handling
  time is berth-independent, which is the special case of its model, so the
  schedule is comparable once the berth and start are read back.
* Yalaoui 2006 solves `P_m | r_j | sum C_j`, which is not a berth allocation
  problem, so it is compared on the parallel-machine instances instead.
"""

from __future__ import annotations

import time

from ..heuristics.constructive import BASELINES
from ..heuristics.metaheuristics import HEURISTICS
from ..instances import Instance, Schedule
from ..metrics import gap_pct, objective_value, sanity_check_schedule
from ..models.bap import BAPTimeIndexed
from .imai2001 import INDIVIDUAL, solve_instance as imai_solve
from .iris2017 import ALNSConfig, run_alns as iris_solve

DEFAULT_SEED = 20261014


def imai_schedule(inst: Instance, procedure: str = INDIVIDUAL,
                  max_iter: int = 200) -> Schedule:
    """The Imai heuristic's best schedule, as a package `Schedule`."""
    result = imai_solve(inst, procedure=procedure, max_iter=max_iter)
    solution = result.best_feasible
    assign: dict[int, tuple[int, int]] = {}
    for i, sequence in enumerate(solution.order):
        # `order[i][k]` is the ship in position k + 1 and `start[i][k]` its start,
        # both indexed from the end of the berth's sequence; the pair is what
        # identifies a vessel's berth and start regardless of that indexing.
        for k, vid in enumerate(sequence):
            assign[vid] = (i, int(solution.start[i][k]))
    if len(assign) != inst.n:
        return Schedule(assign=assign, feasible=False, objective=None,
                        solver=f"Imai2001[{procedure}]",
                        violations=[f"only {len(assign)} of {inst.n} vessels placed"])
    sch = Schedule(assign=assign, solver=f"Imai2001[{procedure}]")
    return sch.check(inst)


def iris_schedule(inst: Instance, objective: str = "sum_C",
                  seed: int = DEFAULT_SEED,
                  max_iterations: int = 4000) -> Schedule:
    """The Iris ALNS schedule, as a package `Schedule`."""
    return iris_solve(inst, objective, seed=seed,
                      config=ALNSConfig(max_iterations=max_iterations))


def _rows_for(inst: Instance, objective: str, methods: dict, time_limit: float,
              reference: float | None, ref_kind: str) -> list[dict]:
    rows = []
    for name, runner in methods.items():
        t0 = time.perf_counter()
        try:
            sch = runner()
        except Exception as exc:                                   # noqa: BLE001
            rows.append({"instance": inst.name, "method": name, "objective": None,
                         "gap_pct": None, "feasible": False,
                         "violations": f"{type(exc).__name__}: {exc}",
                         "wall_time_s": round(time.perf_counter() - t0, 4),
                         "reference": reference, "reference_kind": ref_kind,
                         "origin": "reproduction"})
            continue
        wall = time.perf_counter() - t0
        # Recompute from the assignment rather than trusting the solver's figure,
        # and re-check feasibility with code the method does not use.
        ok, value, problems = False, None, []
        if sch.assign and len(sch.assign) == inst.n:
            checked = sch.check(inst)
            # Two independent re-checks, as the module promises: `Schedule.check`
            # and `sanity_check_schedule` share no code with each other or with
            # any of the methods, so a schedule only counts when both accept it.
            # Calling only the first left the second documented but never run.
            issues = sanity_check_schedule(inst, sch)
            problems = list(checked.violations) + [
                f"{issue.kind}: {issue.detail}" for issue in issues]
            ok = checked.feasible and not issues
            if ok:
                value = objective_value(sch.completion(inst), objective)
        else:
            problems = ["incomplete assignment"]
        gap = None
        if ok and value is not None and reference:
            gap = gap_pct(value, reference)
        rows.append({"instance": inst.name, "method": name, "objective": value,
                     "gap_pct": gap, "feasible": ok,
                     "violations": "; ".join(problems[:2]),
                     "wall_time_s": round(wall, 4),
                     "reference": reference, "reference_kind": ref_kind,
                     "origin": "reproduction"})
    return rows


def compare_reproductions(inst: Instance, objective: str = "sum_C",
                          time_limit: float = 60.0,
                          seed: int = DEFAULT_SEED,
                          procedures: tuple[str, ...] = ("SIMPLE", "INDIVIDUAL",
                                                         "INTERACT"),
                          with_package_methods: bool = True,
                          with_iris: bool = True,
                          imai_max_iter: int = 200) -> list[dict]:
    """Every reproduced method and, optionally, the package's own, on one instance.

    The reference is the exact model's optimum when it proves one, because that is
    the only value all the methods can be scored against; the column says which.

    Expect the Imai procedures to fail on an instance with tidal windows or a crane
    limit: the DBAP it solves has neither, so a schedule that is feasible for its
    own model need not be feasible here. Those rows carry the violation rather than
    being dropped, because that is the comparison's result.
    """
    exact = BAPTimeIndexed(inst, objective=objective,
                           time_limit=time_limit).solve()
    proved = bool(exact.extra.get("proved_optimal")) and exact.objective is not None
    reference = float(exact.objective) if proved else None
    ref_kind = "exact_proved" if proved else "none"

    methods: dict = {}
    for procedure in procedures:
        methods[f"Imai2001[{procedure}]"] = (
            lambda p=procedure: imai_schedule(inst, procedure=p,
                                              max_iter=imai_max_iter))
    if with_iris:
        methods["Iris2017[ALNS]"] = (
            lambda: iris_schedule(inst, objective, seed=seed))
    if with_package_methods:
        for name, fn in BASELINES.items():
            methods[f"package[{name}]"] = (
                lambda f=fn: f(inst, objective))
        for name, fn in HEURISTICS.items():
            methods[f"package[{name}]"] = (
                lambda f=fn: f(inst, objective, seed))

    rows = _rows_for(inst, objective, methods, time_limit, reference, ref_kind)
    if proved:
        rows.insert(0, {"instance": inst.name, "method": "MILP",
                        "objective": reference, "gap_pct": 0.0, "feasible": True,
                        "violations": "", "wall_time_s": round(exact.wall_time, 4),
                        "reference": reference, "reference_kind": ref_kind,
                        "origin": "exact"})
    return rows
