"""Classical scheduling experiments: single machine, parallel machines, flow shop, job shop.

Each group compares a closed-form rule or an MILP against an independent
enumeration, so agreement is evidence rather than self-consistency.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from ..models.classical import (
    FlowShopMILP,
    JobShopMILP,
    ParallelMachineMILP,
    brute_force_parallel,
    brute_force_single_machine,
    make_simple_problem,
    makespan_upper_bound,
    single_machine_spt,
    single_machine_wspt,
    to_instance_view,
)
from ..verification.jobshop import job_shop_optimum


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


def _pfsp_makespan(proc: list[list[int]], order: list[int]) -> int:
    """Makespan of a permutation flow shop, machine by machine."""
    n, m = len(order), len(proc[0])
    done = [0] * m
    for j in order:
        t = 0
        for k in range(m):
            start = max(t, done[k])
            t = start + proc[j][k]
            done[k] = t
    return max(done)


# --------------------------------------------------------------------------
def single_machine(sizes: list[int], out: Path) -> list[dict]:
    """SPT and WSPT against exhaustive enumeration."""
    rows = []
    for n in sizes:
        P = make_simple_problem(f"sm_{n}", n, 1, seed=17 + n, horizon=200)
        _, spt_val = single_machine_spt(P)
        _, wspt_val = single_machine_wspt(P)
        bf_val, bf_order = brute_force_single_machine(P, "sum_C")
        rows.append({
            "n": n,
            "processing_times": " ".join(str(j.proc) for j in P.jobs),
            "spt_sum_C": spt_val,
            "wspt_sum_wC": wspt_val,
            "brute_sum_C": bf_val,
            "brute_order": " ".join(str(j) for j in (bf_order or [])),
            "spt_matches_brute": (None if bf_val is None else spt_val == bf_val),
            "permutations": _factorial(n),
        })
    _write_csv(out / "W3a_single_machine.csv", rows)
    return rows


def _factorial(n: int) -> int:
    total = 1
    for i in range(2, n + 1):
        total *= i
    return total


def single_machine_release(sizes: list[int], out: Path) -> list[dict]:
    """With release dates SPT is no longer optimal; quantify by how much."""
    rows = []
    for n in sizes:
        P = make_simple_problem(f"smr_{n}", n, 1, seed=23 + n, release=True,
                                release_range=(0, 12), horizon=200)
        _, spt_val = single_machine_spt(P)
        bf_val, _ = brute_force_single_machine(P, "sum_C")
        bf_w, _ = brute_force_single_machine_weighted(P)
        _, wspt_val = single_machine_wspt(P)
        rows.append({
            "n": n,
            "releases": " ".join(str(j.release) for j in P.jobs),
            "processing_times": " ".join(str(j.proc) for j in P.jobs),
            "weights": " ".join(str(j.weight) for j in P.jobs),
            "spt_sum_C": spt_val,
            "brute_sum_C": bf_val,
            "spt_gap_pct": _gap(spt_val, bf_val),
            "spt_optimal": spt_val == bf_val,
            "wspt_sum_wC": wspt_val,
            "brute_sum_wC": bf_w,
            "wspt_gap_pct": _gap(wspt_val, bf_w),
            "wspt_optimal": wspt_val == bf_w,
        })
    _write_csv(out / "W3b_single_machine_release.csv", rows)
    return rows


def _gap(value: float | None, optimum: float | None) -> float | None:
    if value is None or optimum is None or optimum == 0:
        return None
    return round((value - optimum) / optimum * 100.0, 3)


def brute_force_single_machine_weighted(prob) -> tuple[float | None, list[int] | None]:
    """Exhaustive minimisation of sum w_j C_j, honouring release dates."""
    from itertools import permutations
    if prob.n > 8:
        return None, None
    best, best_order = None, None
    for order in permutations(range(prob.n)):
        t, total = 0, 0.0
        for j in order:
            job = prob.jobs[j]
            t = max(t, job.release) + job.proc
            total += job.weight * t
        if best is None or total < best:
            best, best_order = total, list(order)
    return best, best_order


def parallel_machines(configs: list[tuple[int, int]], out: Path) -> list[dict]:
    """Parallel machine MILP against enumeration, with and without coupling."""
    rows = []
    variants = [
        ("plain", dict()),
        ("release", dict(release=True)),
        ("unavailable", dict(unavailable=True)),
        ("release+unavailable", dict(release=True, unavailable=True)),
    ]
    for n, m in configs:
        for label, kw in variants:
            horizon = n * 12 + 40
            P = make_simple_problem(f"pm_{n}x{m}_{label}", n, m, seed=31 + n,
                                    horizon=horizon, **kw)
            inst = to_instance_view(P)
            t0 = time.perf_counter()
            sch = ParallelMachineMILP(P, instance_view=inst).solve()
            wall = time.perf_counter() - t0
            bf = brute_force_parallel(P, "sum_C", max_cases=400_000)
            agree = (sch.feasible and bf is not None
                     and abs(sch.objective - bf) < 1e-6)
            rows.append({
                "n": n, "m": m, "variant": label,
                "machines_available": m,
                "milp_sum_C": sch.objective, "milp_feasible": sch.feasible,
                "brute_sum_C": bf,
                "agree": agree,
                "milp_time_s": round(wall, 4),
                "horizon": P.horizon,
                "violations": "; ".join(sch.violations[:2]),
            })
    _write_csv(out / "W3c_parallel_machines.csv", rows)
    return rows


def flow_shop(configs: list[tuple[int, int]], out: Path) -> list[dict]:
    """Permutation flow shop MILP against all permutations."""
    rows = []
    for n, m in configs:
        P = make_simple_problem(f"fs_{n}x{m}", n, m, seed=41 + n, horizon=n * 12 + 40)
        sch = FlowShopMILP(P, instance_view=to_instance_view(P)).solve()
        proc = [[j.proc for _ in range(m)] for j in P.jobs]
        best = None
        for order in _permutations(list(range(n))):
            ms = _pfsp_makespan(proc, order)
            if best is None or ms < best:
                best = ms
        upper = makespan_upper_bound(P)
        rows.append({
            "n": n, "m": m,
            "milp_C_max": sch.objective, "milp_feasible": sch.feasible,
            "enumerated_C_max": best,
            "agree": sch.feasible and best is not None and sch.objective == best,
            "makespan_upper_bound": upper,
            "upper_bound_valid": upper >= (best or 0),
            "permutations": _factorial(n),
        })
    _write_csv(out / "W3d_flow_shop.csv", rows)
    return rows


def _permutations(items: list[int]):
    if len(items) <= 1:
        yield list(items)
        return
    for i, x in enumerate(items):
        rest = items[:i] + items[i + 1:]
        for p in _permutations(rest):
            yield [x, *p]


def job_shop(configs: list[tuple[int, int]], out: Path,
             time_budget: float = 60.0) -> list[dict]:
    """Job shop MILP against an exact branch-and-bound search that uses no solver."""
    rows = []
    for n, m in configs:
        P = make_simple_problem(f"js_{n}x{m}", n, m, seed=51 + n, horizon=n * 12 + 40)
        routes = {j: list(range(P.m)) for j in range(P.n)}
        inst = to_instance_view(P)
        sch = JobShopMILP(P, routes=routes, instance_view=inst).solve()
        proc = [[j.proc for _ in range(m)] for j in P.jobs]
        # reference schedule: jobs in index order, each started as early as possible
        ref = _pfsp_makespan(proc, list(range(n)))

        search = job_shop_optimum([list(range(m)) for _ in range(n)], proc,
                                  time_budget=time_budget)
        agree = (sch.feasible and search.objective is not None
                 and abs(sch.objective - search.objective) < 1e-6)
        rows.append({
            "n": n, "m": m,
            "milp_C_max": sch.objective, "milp_feasible": sch.feasible,
            "independent_C_max": search.objective,
            "agree_with_independent": agree if search.objective is not None else None,
            "search_nodes": search.nodes,
            "search_time_s": round(search.wall_time, 4),
            "search_note": search.note,
            "reference_C_max": ref,
            "not_worse_than_reference": (sch.feasible and sch.objective is not None
                                         and sch.objective <= ref),
            "upper_bound": makespan_upper_bound(P),
            "violations": "; ".join(sch.violations[:2]),
        })
    _write_csv(out / "W3e_job_shop.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="bapbench classical",
        description="Classical scheduling experiments against independent enumeration.")
    p.add_argument("--out", type=Path, default=Path("results/classical"))
    p.add_argument("--single-sizes", type=int, nargs="+", default=[4, 5, 6, 7])
    p.add_argument("--release-sizes", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--parallel", type=int, nargs="+", default=[5, 3],
                   help="n and m, given as two numbers")
    p.add_argument("--flow", type=int, nargs="+", default=[4, 3])
    p.add_argument("--jobshop", type=int, nargs="+", default=[4, 3])
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["single", "release", "parallel", "flow", "jobshop"])
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if "single" not in args.skip:
        rows = single_machine(args.single_sizes, out)
        print("Single machine: SPT and WSPT against enumeration")
        print(f"  {'n':>3} {'SPT':>8} {'WSPT(wC)':>10} {'brute':>8} {'match':>7}")
        for r in rows:
            print(f"  {r['n']:>3} {r['spt_sum_C']:>8} {r['wspt_sum_wC']:>10} "
                  f"{str(r['brute_sum_C']):>8} {str(r['spt_matches_brute']):>7}")
        print(f"  wrote {out / 'W3a_single_machine.csv'}\n")

    if "release" not in args.skip:
        rows = single_machine_release(args.release_sizes, out)
        print("Single machine with release dates: SPT is no longer optimal")
        print(f"  {'n':>3} {'SPT':>8} {'brute':>8} {'gap %':>9} {'opt':>6} "
              f"{'WSPT':>8} {'gap %':>9} {'opt':>6}")
        for r in rows:
            print(f"  {r['n']:>3} {r['spt_sum_C']:>8} {str(r['brute_sum_C']):>8} "
                  f"{str(r['spt_gap_pct']):>9} {str(r['spt_optimal']):>6} "
                  f"{r['wspt_sum_wC']:>8} {str(r['wspt_gap_pct']):>9} "
                  f"{str(r['wspt_optimal']):>6}")
        print(f"  wrote {out / 'W3b_single_machine_release.csv'}\n")

    if "parallel" not in args.skip:
        cfg = [(args.parallel[0], args.parallel[1])]
        rows = parallel_machines(cfg, out)
        print("Parallel machines: MILP against enumeration")
        print(f"  {'variant':<22} {'MILP':>8} {'brute':>8} {'agree':>7}")
        for r in rows:
            print(f"  {r['variant']:<22} {str(r['milp_sum_C']):>8} "
                  f"{str(r['brute_sum_C']):>8} {str(r['agree']):>7}")
        print(f"  wrote {out / 'W3c_parallel_machines.csv'}\n")

    if "flow" not in args.skip:
        cfg = [(args.flow[0], args.flow[1])]
        rows = flow_shop(cfg, out)
        print("Permutation flow shop: MILP against all permutations")
        print(f"  {'n':>3} {'m':>3} {'MILP':>8} {'enum':>8} {'agree':>7} {'UB ok':>7}")
        for r in rows:
            print(f"  {r['n']:>3} {r['m']:>3} {str(r['milp_C_max']):>8} "
                  f"{str(r['enumerated_C_max']):>8} {str(r['agree']):>7} "
                  f"{str(r['upper_bound_valid']):>7}")
        print(f"  wrote {out / 'W3d_flow_shop.csv'}\n")

    if "jobshop" not in args.skip:
        cfg = [(args.jobshop[0], args.jobshop[1])]
        rows = job_shop(cfg, out)
        print("Job shop: MILP against an exact branch-and-bound search")
        print(f"  {'n':>3} {'m':>3} {'MILP':>8} {'independent':>12} "
              f"{'agree':>7} {'nodes':>9} {'time s':>8}")
        for r in rows:
            print(f"  {r['n']:>3} {r['m']:>3} {str(r['milp_C_max']):>8} "
                  f"{str(r['independent_C_max']):>12} "
                  f"{str(r['agree_with_independent']):>7} "
                  f"{r['search_nodes']:>9} {r['search_time_s']:>8}")
        print(f"  wrote {out / 'W3e_job_shop.csv'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
