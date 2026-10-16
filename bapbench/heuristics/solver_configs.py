# -*- coding: utf-8 -*-
"""Solver configuration comparison.

Only one solver backend is available, so the comparison runs several
configurations of it: gap tolerance, time limit and presolve. Configurations
are plain data passed as solver keyword arguments, keeping backend option
names out of the model class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..instances import Instance, Schedule
from ..models.bap import BAPTimeIndexed


@dataclass
class SolverConfig:
    name: str
    kwargs: dict
    note: str = ""


CONFIGS: list[SolverConfig] = [
    SolverConfig("default", {}, "solver defaults"),
    SolverConfig("tightgap", {"gapRel": 0.0}, "gapRel = 0"),
    SolverConfig("shorttime", {"timeLimit": 2.0}, "2 s limit, as an online budget"),
    SolverConfig("nopresolve", {"presolve": "off"}, "presolve disabled"),
    SolverConfig("tight+short", {"gapRel": 0.0, "timeLimit": 2.0},
                 "gapRel = 0 with a 2 s limit"),
]


def solve_with_config(inst: Instance, cfg: SolverConfig,
                      objective: str = "sum_C") -> Schedule:
    """Solve the BAP under one configuration; options go through solver_kwargs.
    A time-limited run may be feasible without being proven optimal.
    """
    t0 = time.perf_counter()
    model = BAPTimeIndexed(inst, objective=objective,
                           solver_kwargs=dict(cfg.kwargs))
    sch = model.solve()
    wall = time.perf_counter() - t0
    sch.solver = f"HiGHS[{cfg.name}]"
    sch.wall_time = wall
    sch.extra = {**(sch.extra or {}), "config": cfg.name, "config_note": cfg.note,
                 "config_kwargs": dict(cfg.kwargs)}
    return sch


def run_config_comparison(inst: Instance, objective: str = "sum_C",
                          reference: float | None = None) -> list[dict]:
    """Evaluate every configuration; reference is a known optimum, or None."""
    rows: list[dict] = []
    for cfg in CONFIGS:
        try:
            sch = solve_with_config(inst, cfg, objective)
        except Exception as e:                                   # noqa: BLE001
            rows.append({"config": cfg.name, "note": cfg.note, "objective": None,
                         "feasible": False, "wall_time_s": None,
                         "proved_optimal": None, "gap_pct": None,
                         "vars": None, "cons": None,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        # Optimality is claimed only when the solver reported it, never inferred
        # from the configuration: a run with gapRel=0 and a time limit can stop
        # on the limit and still look "exact" by its own settings.
        proved = bool(sch.extra.get("proved_optimal")) and sch.feasible
        gap = None
        if reference is not None and sch.objective is not None:
            gap = round((sch.objective - reference) / reference * 100.0, 3) \
                if reference else None
        rows.append({
            "config": cfg.name, "note": cfg.note,
            "objective": sch.objective, "feasible": sch.feasible,
            "wall_time_s": round(sch.wall_time, 4),
            "proved_optimal": proved,
            "solver_status": (sch.extra or {}).get("status"),
            "gap_pct": gap,
            "vars": (sch.extra or {}).get("vars"),
            "cons": (sch.extra or {}).get("cons"),
            "error": "",
        })
    return rows
