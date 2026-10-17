"""Metaheuristic experiments, including repeated runs across seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from ..bounds import reference_for
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
    """Run one method. Raises only for an unknown method name.

    The greedy decoder cannot place every vessel on every instance -- an
    unfavourable order can strand the last ones -- so a failed placement is
    returned as ``(None, wall, False, reason)`` and recorded, not raised: a
    method that cannot build a schedule is a result, not a crash.
    """
    t0 = time.perf_counter()
    try:
        if method == "MILP":
            sch = BAPTimeIndexed(inst, objective=objective, time_limit=time_limit,
                                 solver_kwargs={"gapRel": 0.0}).solve()
        elif method in BASELINES:
            fn = BASELINES[method]
            sch = (fn(inst, objective, seed=seed) if method in ("SA", "GA")
                   else fn(inst, objective))
        elif method in HEURISTICS:
            sch = HEURISTICS[method](inst, objective, seed)
        else:
            raise KeyError(method)
    except RuntimeError as exc:
        return None, time.perf_counter() - t0, False, str(exc)[:160]
    return sch, None, None, None


def _method_names() -> list[str]:
    return ["MILP"] + list(BASELINES) + list(HEURISTICS)


def run_seeds(inst, objective: str, n_seeds: int, time_limit: float,
              with_bound: bool = True) -> list[dict]:
    exact = BAPTimeIndexed(inst, objective=objective, time_limit=time_limit,
                           solver_kwargs={"gapRel": 0.0}).solve()
    proved = bool(exact.extra.get("proved_optimal")) and exact.objective is not None
    optimum = exact.objective if proved else None

    # An unproved exact value is an upper bound on the optimum, so using it as a
    # reference would understate every gap. reference_for falls back to the
    # strongest valid bound and labels which reference was used.
    if with_bound:
        reference, ref_kind = reference_for(inst, objective, exact.objective,
                                            proved, time_limit=30.0)
    else:
        reference, ref_kind = (float(exact.objective), "exact_proved") if proved \
            else (None, "none")

    names = _method_names()
    buckets: dict[str, list[tuple]] = {n: [] for n in names}
    reasons: dict[str, list[str]] = {n: [] for n in names}
    # The MILP was solved above to obtain the reference and is deterministic, so
    # solving it once per seed repeated identical work `n_seeds` times (once per
    # seed plus the reference solve) and could only differ across seeds if the
    # solver were nondeterministic -- which the determinism self-check would then
    # report against the wrong method. Its row reuses that one run.
    milp_ok = bool(exact.feasible and exact.objective is not None)
    milp_entry = (exact.objective if milp_ok else None, exact.wall_time, milp_ok)
    for _seed in range(n_seeds):
        buckets["MILP"].append(milp_entry)
        if not milp_ok:
            reasons["MILP"].append("; ".join(exact.violations[:1]) or "infeasible")
        for name in names:
            if name == "MILP":
                continue
            sch, wall, _n, reason = _run_one(inst, name, objective, _seed, time_limit)
            if sch is None:
                buckets[name].append((None, wall, False))
                reasons[name].append(reason or "placement failed")
                continue
            ok = bool(sch.feasible and sch.objective is not None)
            buckets[name].append((sch.objective if ok else None, sch.wall_time, ok))
            if not ok:
                reasons[name].append("; ".join(sch.violations[:1]) or "infeasible")

    rows = []
    for name in names:
        vals = [v for (v, _t, ok) in buckets[name] if ok and v is not None]
        times = [t for (_v, t, _ok) in buckets[name]]
        # "Hit the reference" is only meaningful against a proved optimum. Against a
        # lower bound a method can be strictly better than the reference and still
        # not equal it, so counting equality would report 0 hits by construction --
        # and the bound is the reference in the regime this experiment targets. The
        # count is left empty rather than zero when there is nothing to compare to.
        hits = None
        if ref_kind == "exact_proved" and reference is not None:
            hits = sum(1 for v in vals if abs(v - reference) < 1e-6)
        mean = statistics.mean(vals) if vals else None
        rows.append({
            "instance": inst.name, "slack": inst.meta.get("slack"), "method": name,
            "kind": "deterministic" if name in DETERMINISTIC else "stochastic",
            "n_seeds": n_seeds, "n_valid": len(vals),
            "n_failed": sum(1 for (_v, _t, ok) in buckets[name] if not ok),
            "failure_reason": reasons[name][0] if reasons[name] else "",
            "mean_objective": round(mean, 4) if mean is not None else None,
            "std_objective": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min_objective": min(vals) if vals else None,
            "max_objective": max(vals) if vals else None,
            "reference": round(reference, 4) if reference is not None else None,
            "reference_kind": ref_kind,
            "exact": optimum,
            "mean_gap_pct": gap_pct(mean, reference) if mean is not None else None,
            "best_gap_pct": gap_pct(min(vals), reference) if vals else None,
            "worst_gap_pct": gap_pct(max(vals), reference) if vals else None,
            "hit_optimal": hits,
            "hit_optimal_pct": (round(hits / n_seeds * 100.0, 1)
                                if hits is not None else None),
            "mean_wall_time_s": round(statistics.mean(times), 4) if times else None,
        })
    return rows


def self_check(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        # A spread of zero is only evidence of determinism when at least two runs
        # produced a value; with fewer, the spread is zero by definition and saying
        # nothing would let a method that failed every seed pass the check.
        if r["kind"] == "deterministic":
            if r.get("n_valid", 0) < 2:
                problems.append(
                    f"{r['method']} is deterministic but only "
                    f"{r.get('n_valid', 0)} of {r.get('n_seeds', '?')} runs produced "
                    f"a value, so its spread cannot show whether it varies")
            elif r["std_objective"] not in (0, 0.0):
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
        ref = rows[0]["reference"] if rows else None
        ref_kind = rows[0]["reference_kind"] if rows else "none"
        print(f"  reference = {ref} ({ref_kind})")
        print(f"  {'method':<10} {'kind':<14} {'mean':>9} {'sd':>8} {'best':>8} "
              f"{'worst':>8} {'gap %':>8} {'hit ref':>9}")
        for r in rows:
            hit = ("n/a" if r["hit_optimal"] is None
                   else f"{r['hit_optimal']}/{r['n_seeds']}")
            print(f"  {r['method']:<10} {r['kind']:<14} {str(r['mean_objective']):>9} "
                  f"{r['std_objective']:>8} {str(r['min_objective']):>8} "
                  f"{str(r['max_objective']):>8} {str(r['mean_gap_pct']):>8} "
                  f"{hit:>9}")
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
          f"{'gap %':>8} {'ref kind':>12} {'exact s':>9}")
    for n in args.sizes:
        inst = make_toy_instance(f"scale_{n}x{args.berths}", n, args.berths,
                                 seed=100 + n, tight=True)
        r = run_seeds(inst, args.objective, args.seeds, args.time_limit)
        reference = r[0]["reference"] if r else None
        ref_kind = r[0]["reference_kind"] if r else "none"
        exact = r[0]["exact"] if r else None
        proved = ref_kind == "exact_proved"
        milp = next((x for x in r if x["method"] == "MILP"), None)
        heur = [x for x in r
                if x["method"] != "MILP" and x["min_objective"] is not None]
        best = min(heur, key=lambda x: x["min_objective"]) if heur else None
        rows.append({
            "n": n, "m": args.berths, "exact_obj": exact,
            "proved": proved,
            "exact_time_s": milp["mean_wall_time_s"] if milp else None,
            "reference": reference, "reference_kind": ref_kind,
            "best_heuristic": best["min_objective"] if best else None,
            "best_heuristic_name": best["method"] if best else None,
            "gap_pct": (gap_pct(best["min_objective"], reference)
                        if best and reference is not None else None),
        })
        print(f"  {n:>3} {args.berths:>3} {str(exact):>9} "
              f"{str(proved):>7} {str(rows[-1]['best_heuristic_name']):>10} "
              f"{str(rows[-1]['gap_pct']):>8} {ref_kind:>12} {'':>9}")
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
    p = argparse.ArgumentParser(prog="bapbench heuristics",
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
