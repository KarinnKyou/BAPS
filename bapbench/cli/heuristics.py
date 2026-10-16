"""Metaheuristic experiments, including repeated runs across seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from ..heuristics.constructive import BASELINES
from ..heuristics.metaheuristics import HEURISTICS
from ..heuristics.solver_configs import run_config_comparison
from ..instances import make_hard_instance, make_toy_instance
from ..metrics import gap_pct
from ..models.bap import BAPTimeIndexed

# Methods without internal randomness. Their objective values must have zero
# spread across seeds; any spread means uncontrolled randomness leaked in.
DETERMINISTIC = {"MILP", "FCFS", "LPT", "SPT", "GREEDY"}

DEFAULT_SEEDS = 10


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _run_one(inst, method: str, objective: str, seed: int, time_limit: float):
    if method == "MILP":
        return BAPTimeIndexed(inst, objective=objective, time_limit=time_limit,
                              solver_kwargs={"gapRel": 0.0}).solve()
    if method in BASELINES:
        fn = BASELINES[method]
        if method in ("SA", "GA"):
            return fn(inst, objective, seed=seed)
        return fn(inst, objective)
    if method in HEURISTICS:
        return HEURISTICS[method](inst, objective, seed)
    raise KeyError(method)


def _method_names() -> list[str]:
    return ["MILP"] + list(BASELINES) + list(HEURISTICS)


def run_seeds(inst, objective: str, n_seeds: int, time_limit: float) -> list[dict]:
    exact = BAPTimeIndexed(inst, objective=objective, time_limit=time_limit,
                           solver_kwargs={"gapRel": 0.0}).solve()
    optimum = exact.objective if exact.feasible else None

    names = _method_names()
    buckets: dict[str, list[tuple]] = {n: [] for n in names}
    for seed in range(n_seeds):
        for name in names:
            sch = _run_one(inst, name, objective, seed, time_limit)
            ok = bool(sch.feasible and sch.objective is not None)
            buckets[name].append((sch.objective if ok else None, sch.wall_time, ok))

    rows = []
    for name in names:
        vals = [v for (v, _t, ok) in buckets[name] if ok and v is not None]
        times = [t for (_v, t, _ok) in buckets[name]]
        hits = sum(1 for v in vals if optimum is not None and abs(v - optimum) < 1e-6)
        mean = statistics.mean(vals) if vals else None
        rows.append({
            "instance": inst.name, "slack": inst.meta.get("slack"), "method": name,
            "kind": "deterministic" if name in DETERMINISTIC else "stochastic",
            "n_seeds": n_seeds, "n_valid": len(vals),
            "n_failed": sum(1 for (_v, _t, ok) in buckets[name] if not ok),
            "mean_objective": round(mean, 4) if mean is not None else None,
            "std_objective": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min_objective": min(vals) if vals else None,
            "max_objective": max(vals) if vals else None,
            "mean_gap_pct": gap_pct(mean, optimum) if mean is not None else None,
            "best_gap_pct": gap_pct(min(vals), optimum) if vals else None,
            "worst_gap_pct": gap_pct(max(vals), optimum) if vals else None,
            "hit_optimal": hits,
            "hit_optimal_pct": round(hits / n_seeds * 100.0, 1),
            "mean_wall_time_s": round(statistics.mean(times), 4) if times else None,
            "optimal": optimum,
        })
    return rows


def self_check(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        if r["kind"] == "deterministic" and r["std_objective"] not in (0, 0.0):
            problems.append(
                f"{r['method']} is deterministic but its objective spread is "
                f"{r['std_objective']}, which means uncontrolled randomness")
    return problems


def cmd_multiseed(args) -> int:
    all_rows: list[dict] = []
    for slack in args.slacks:
        inst = make_hard_instance(f"ms_slack{slack}", args.instances, args.berths,
                                  seed=args.seed, slack=slack)
        print(inst.summary())
        rows = run_seeds(inst, args.objective, args.seeds, args.time_limit)
        all_rows.extend(rows)
        print(f"  {'method':<10} {'kind':<14} {'mean':>9} {'sd':>8} {'best':>8} "
              f"{'worst':>8} {'gap %':>8} {'optimal':>9}")
        for r in rows:
            print(f"  {r['method']:<10} {r['kind']:<14} {str(r['mean_objective']):>9} "
                  f"{r['std_objective']:>8} {str(r['min_objective']):>8} "
                  f"{str(r['max_objective']):>8} {str(r['mean_gap_pct']):>8} "
                  f"{r['hit_optimal']:>4}/{r['n_seeds']:<4}")
        print()

    _write_csv(args.out / "multiseed.csv", all_rows)
    print(f"wrote {args.out / 'multiseed.csv'} ({len(all_rows)} rows)")

    problems = self_check(all_rows)
    if problems:
        print("\nself-check failed:")
        for p in problems:
            print("  -", p)
        return 1
    print("self-check passed: deterministic methods show zero spread")
    return 0


def cmd_scaling(args) -> int:
    rows = []
    print(f"  {'n':>3} {'m':>3} {'exact':>9} {'proved':>7} {'best heur':>10} "
          f"{'gap %':>8} {'exact s':>9}")
    for n in args.sizes:
        inst = make_toy_instance(f"scale_{n}x{args.berths}", n, args.berths,
                                 seed=100 + n, tight=True)
        exact = BAPTimeIndexed(inst, objective=args.objective,
                               time_limit=args.time_limit).solve()
        proved = bool(exact.feasible and exact.objective is not None)
        r = run_seeds(inst, args.objective, args.seeds, args.time_limit)
        heur = [x for x in r if x["method"] != "MILP" and x["min_objective"]]
        best = min(heur, key=lambda x: x["min_objective"]) if heur else None
        rows.append({
            "n": n, "m": args.berths, "exact_obj": exact.objective if proved else None,
            "proved": proved, "exact_time_s": round(exact.wall_time, 4),
            "best_heuristic": best["min_objective"] if best else None,
            "best_heuristic_name": best["method"] if best else None,
            "gap_pct": (gap_pct(best["min_objective"], exact.objective)
                        if best and proved else None),
        })
        print(f"  {n:>3} {args.berths:>3} {str(rows[-1]['exact_obj']):>9} "
              f"{str(proved):>7} {str(rows[-1]['best_heuristic_name']):>10} "
              f"{str(rows[-1]['gap_pct']):>8} {exact.wall_time:>9.4f}")
    _write_csv(args.out / "scaling.csv", rows)
    print(f"\nwrote {args.out / 'scaling.csv'}")
    return 0


def cmd_configs(args) -> int:
    rows = []
    print(f"  {'instance':<24} {'config':<16} {'objective':>10} {'time s':>9}")
    for slack in args.slacks:
        inst = make_hard_instance(f"cfg_slack{slack}", args.instances, args.berths,
                                  seed=args.seed, slack=slack)
        for r in run_config_comparison(inst, args.objective, None):
            r["instance"] = inst.name
            r["slack"] = slack
            rows.append(r)
            print(f"  {inst.name:<24} {str(r.get('config')):<16} "
                  f"{str(r.get('objective')):>10} {str(r.get('wall_time_s')):>9}")
    _write_csv(args.out / "solver_configs.csv", rows)
    print(f"\nwrote {args.out / 'solver_configs.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baps heuristics",
                                description="Metaheuristic and solver-configuration runs.")
    p.add_argument("mode", choices=("multiseed", "scaling", "configs"),
                   nargs="?", default="multiseed")
    p.add_argument("--out", type=Path, default=Path("results/heuristics"))
    p.add_argument("--instances", type=int, default=12)
    p.add_argument("--berths", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--objective", choices=("sum_C", "C_max"), default="sum_C")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--slacks", type=float, nargs="+", default=[1.8, 1.6])
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    args = p.parse_args(argv)

    dispatch = {"multiseed": cmd_multiseed, "scaling": cmd_scaling,
                "configs": cmd_configs}
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
