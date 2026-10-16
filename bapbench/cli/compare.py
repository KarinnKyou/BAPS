"""Exact-versus-heuristic comparisons, constraint ablation and slack scaling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from ..bounds import best_lower_bound, trivial_lower_bound
from ..heuristics.constructive import BASELINES, run_all_baselines
from ..instances import Instance, make_hard_instance, make_toy_instance
from ..metrics import gap_pct
from ..models.bap import BAPTimeIndexed

# Fixed horizon for the ablation. At this slack the exact model still proves
# optimality, so the only variable left is which constraints are switched on.
ABLATION_HORIZON = 76

# slack is stated explicitly for the constrained instance: the generator default of
# 1.0 leaves no feasible schedule once closures and the crane limit apply together,
# so every method including the exact model reports failure.
DEFAULT_INSTANCES = [
    ("small_5x3", dict(n=5, m=3, seed=20261014, tight=False)),
    ("tight_6x3", dict(n=6, m=3, seed=11, tight=True)),
    ("tight_8x4", dict(n=8, m=4, seed=7, tight=True)),
    ("hard_10x3", dict(n=10, m=3, seed=20261014, hard=True, slack=1.8)),
]


def build(spec: dict) -> Instance:
    hard = spec.pop("hard", False)
    n, m = spec.pop("n"), spec.pop("m")
    name = spec.pop("name")
    maker = make_hard_instance if hard else make_toy_instance
    inst = maker(name, n, m, **spec)
    problems = inst.validate()
    if problems:
        raise ValueError(
            f"{name} is infeasible by construction: " + "; ".join(problems) +
            " (raise slack for constrained instances)")
    # Instance.validate only inspects vessels one at a time, so it cannot see a
    # joint crane-capacity shortfall. At these sizes the exact model can.
    probe = BAPTimeIndexed(inst, time_limit=30.0).solve()
    if not probe.feasible:
        raise ValueError(
            f"{name} has no feasible schedule (exact model: "
            f"{'; '.join(probe.violations) or 'infeasible'}); "
            f"raise slack for constrained instances")
    return inst


def solve_exact(inst: Instance, objective: str, time_limit: float,
                solver_kwargs: dict | None = None) -> tuple:
    """Solve exactly, reporting feasibility and proof of optimality separately.

    `proved` must come from the solver, not from the run being feasible: a
    time-limited incumbent is an upper bound on the optimum, so treating it as
    the optimum turns every gap computed against it into a wrong number.
    """
    model = BAPTimeIndexed(inst, objective=objective, time_limit=time_limit,
                           mip_gap=0.0, solver_kwargs=solver_kwargs or {})
    sch = model.solve()
    proved = bool(sch.extra.get("proved_optimal")) and sch.objective is not None
    return sch, proved


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _best_feasible(schedules: dict) -> tuple[str | None, float | None]:
    ok = {k: v for k, v in schedules.items()
          if v.feasible and v.objective is not None}
    if not ok:
        return None, None
    name = min(ok, key=lambda k: ok[k].objective)
    return name, ok[name].objective


def compare(inst: Instance, objective: str, time_limit: float) -> list[dict]:
    rows: list[dict] = []
    exact, proved = solve_exact(inst, objective, time_limit)
    optimum = exact.objective if proved else None

    # With no proved optimum the gap columns would be empty. A valid lower bound
    # keeps the run reportable, provided the column says which reference it is.
    if not proved:
        bound = best_lower_bound(inst, objective, time_limit=30.0)
        reference, ref_kind = bound.value, f"{bound.kind}_bound"
    else:
        reference, ref_kind = optimum, "exact_proved"

    row = {"instance": inst.name, "n": inst.n, "m": inst.m, "slack": inst.meta.get("slack"),
           "method": "MILP", "objective": optimum, "gap_pct": 0.0 if proved else None,
           "feasible": exact.feasible, "proved_optimal": proved,
           "wall_time_s": round(exact.wall_time, 4), "sum_waiting": None,
           "C_max": None, "berth_util": None,
           "reference": reference, "reference_kind": ref_kind,
           "note": "" if proved else "optimality not proved within the time limit"}
    if proved:
        mt = exact.metrics(inst)
        row.update(sum_waiting=mt["sum_waiting"], C_max=mt["C_max"],
                   berth_util=round(mt["berth_utilization"], 4))
    rows.append(row)

    for name, sch in run_all_baselines(inst, objective).items():
        gap = gap_pct(sch.objective, reference) if sch.feasible else None
        rec = {"instance": inst.name, "n": inst.n, "m": inst.m,
               "slack": inst.meta.get("slack"), "method": name,
               "objective": sch.objective, "gap_pct": gap, "feasible": sch.feasible,
               "proved_optimal": False, "wall_time_s": round(sch.wall_time, 5),
               "sum_waiting": None, "C_max": None, "berth_util": None,
               "reference": reference, "reference_kind": ref_kind,
               "note": "" if sch.feasible else "; ".join(sch.violations)}
        if sch.feasible:
            mt = sch.metrics(inst)
            rec.update(sum_waiting=mt["sum_waiting"], C_max=mt["C_max"],
                       berth_util=round(mt["berth_utilization"], 4))
        rows.append(rec)
    return rows


def ablation(instances_dir: Path, out_dir: Path, objective: str,
             time_limit: float) -> list[dict]:
    configs = [
        ("no_constraints", dict(tidal=False, crane_limit=None)),
        ("tidal_only", dict(tidal=True, crane_limit=None)),
        ("cranes_only", dict(tidal=False, crane_limit=2)),
        ("tidal_and_cranes", dict(tidal=True, crane_limit=2)),
    ]
    rows: list[dict] = []
    for label, cfg in configs:
        inst = make_hard_instance(f"ablation_{label}", 10, 3,
                                  horizon=ABLATION_HORIZON, seed=20261014, **cfg)
        exact, proved = solve_exact(inst, objective, time_limit)
        if not proved:
            rows.append({"config": label, "exact": exact.objective, "best_heuristic": None,
                         "best_heuristic_name": None, "gap_pct": None,
                         "exact_time_s": round(exact.wall_time, 4),
                         "note": "optimality not proved"})
            continue
        name, value = _best_feasible(run_all_baselines(inst, objective))
        rows.append({
            "config": label, "exact": exact.objective, "best_heuristic": value,
            "best_heuristic_name": name,
            "gap_pct": gap_pct(value, exact.objective),
            "exact_time_s": round(exact.wall_time, 4),
            "note": "" if name else "no feasible heuristic",
        })
    return rows


def slack_scan(out_dir: Path, objective: str, time_limit: float,
               slacks: list[float]) -> list[dict]:
    rows: list[dict] = []
    for slack in slacks:
        inst = make_hard_instance(f"slack_{slack}", 10, 3, seed=20261014, slack=slack)
        exact, proved = solve_exact(inst, objective, time_limit)
        status = str(exact.extra.get("status", ""))
        lower_bound = trivial_lower_bound(inst, objective)
        base = {"slack": slack, "horizon": inst.horizon,
                "horizon_over_lower_bound": inst.meta.get("horizon_over_lower_bound"),
                "exact_time_s": round(exact.wall_time, 4)}

        if not proved and not exact.feasible:
            # Three different situations, previously all reported as a timeout.
            # "Infeasible" means the model proved no schedule exists, which is a
            # fact about the instance, not a failure of the solver.
            note = ("instance is infeasible" if "Infeasible" in status
                    else f"exact model stopped with status {status}")
            rows.append({**base, "exact": None, "reference": None,
                         "reference_kind": "none", "best_heuristic": None,
                         "best_heuristic_name": None, "gap_pct": None,
                         "fcfs_gap_pct": None, "lpt_gap_pct": None, "note": note})
            continue

        # With no proved optimum, report the gap against a valid bound instead of
        # leaving the row empty.
        if proved:
            reference, ref_kind = exact.objective, "exact_proved"
        else:
            bound = best_lower_bound(inst, objective, time_limit=30.0)
            reference, ref_kind = bound.value, f"{bound.kind}_bound"

        schedules = run_all_baselines(inst, objective)
        name, value = _best_feasible(schedules)

        def gj(method: str) -> float | None:
            sch = schedules.get(method)
            if sch is None or not sch.feasible or reference is None:
                return None
            return gap_pct(sch.objective, reference)

        rows.append({**base, "exact": exact.objective if proved else None,
                     "reference": reference, "reference_kind": ref_kind,
                     "trivial_bound": lower_bound.value,
                     "best_heuristic": value, "best_heuristic_name": name,
                     "gap_pct": gap_pct(value, reference) if reference else None,
                     "fcfs_gap_pct": gj("FCFS"), "lpt_gap_pct": gj("LPT"),
                     "note": "" if proved else "gap measured against a lower bound"})
    return rows


def scale_scan(out_dir: Path, objective: str, time_limit: float,
               sizes: list[int], n_berths: int = 3) -> list[dict]:
    rows: list[dict] = []
    for n in sizes:
        inst = make_toy_instance(f"scale_{n}x{n_berths}", n, n_berths,
                                 seed=100 + n, tight=True)
        exact, proved = solve_exact(inst, objective, time_limit)
        name, value = _best_feasible(run_all_baselines(inst, objective))
        rows.append({
            "n": n, "m": n_berths, "exact_obj": exact.objective if proved else None,
            "proved": proved, "exact_time_s": round(exact.wall_time, 4),
            "best_heur": value, "best_heur_name": name,
            "gap_pct": gap_pct(value, exact.objective) if proved else None,
            "note": "" if proved else "optimality not proved",
        })
    return rows


def _print_table(rows: list[dict], cols: list[str]) -> None:
    print("  " + "  ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print("  " + "  ".join(f"{str(r.get(c, '')):>14}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bapbench compare",
                                description="Compare the exact model with the baselines.")
    p.add_argument("--out", type=Path, default=Path("results/compare"))
    p.add_argument("--objective", choices=("sum_C", "C_max"), default="sum_C")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--slacks", type=float, nargs="+",
                   default=[3.0, 2.5, 2.0, 1.75, 1.6, 1.5])
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 8, 10, 12])
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["fixed", "ablation", "slack", "scale"])
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if "fixed" not in args.skip:
        print("Fixed instances: exact model versus all baselines")
        rows: list[dict] = []
        for name, spec in DEFAULT_INSTANCES:
            spec = dict(spec, name=name)
            inst = build(spec)
            rows.extend(compare(inst, args.objective, args.time_limit))
        _write_csv(out / "baselines.csv", rows)
        _print_table(rows, ["instance", "method", "objective", "gap_pct"])
        print(f"wrote {out / 'baselines.csv'} ({len(rows)} rows)\n")

    if "ablation" not in args.skip:
        print(f"Constraint ablation at fixed horizon {ABLATION_HORIZON}")
        rows = ablation(out, out, args.objective, args.time_limit)
        _write_csv(out / "ablation.csv", rows)
        _print_table(rows, ["config", "exact", "best_heuristic", "gap_pct"])
        print(f"wrote {out / 'ablation.csv'}\n")

    if "slack" not in args.skip:
        print("Slack scan")
        rows = slack_scan(out, args.objective, args.time_limit, args.slacks)
        _write_csv(out / "slack.csv", rows)
        _print_table(rows, ["slack", "horizon", "exact", "best_heuristic", "gap_pct"])
        print(f"wrote {out / 'slack.csv'}\n")

    if "scale" not in args.skip:
        print("Size scan")
        rows = scale_scan(out, args.objective, args.time_limit, args.sizes)
        _write_csv(out / "scaling.csv", rows)
        _print_table(rows, ["n", "m", "exact_obj", "proved", "best_heur", "gap_pct"])
        print(f"wrote {out / 'scaling.csv'}\n")

    (out / "summary.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": args.objective,
        "objective_definition": "sum_i C_i in periods" if args.objective == "sum_C"
                                else "max_i C_i in periods",
        "dt_hours": 0.5,
        "gap_definition": "(heuristic - optimal) / optimal * 100",
        "baselines": list(BASELINES),
        "time_limit_s": args.time_limit,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
